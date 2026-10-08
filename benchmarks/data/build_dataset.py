"""Deterministically build the synthetic benchmark conversation and questions.

The data is synthetic and was written by an LLM. Signal messages are phrased in
natural conversational language (not shaped around the heuristic extractor's
patterns). Distractors reuse the same entity names as the signals, so the graph
has realistic noisy hubs. Run this script to regenerate conversation.json and
questions.json; both are committed so the gold labels can be reviewed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from ralc.allocation.budget import DefaultTokenCounter

HERE = Path(__file__).resolve().parent
TOKEN_TARGET = 21000

# --------------------------------------------------------------------------
# Signal messages: the facts the questions are about. Natural language.
# --------------------------------------------------------------------------
SIGNALS: dict[str, tuple[str, str]] = {
    "db_mongo_plan": ("user",
        "For the first cut I think we just keep everything in MongoDB. It is quick "
        "to get going, we can throw documents at it, and we do not have to think "
        "about migrations while the shape of an invoice is still changing week to week."),
    "db_mongo_ack": ("assistant",
        "That is fine for the prototype. Once the data model settles down we should "
        "look at it again, but there is no point agonizing over storage this early."),
    "db_postgres_switch": ("user",
        "I have changed my mind about where we keep our data. Invoices have to be "
        "exactly right, and we keep running into places where several rows need to "
        "change together or not at all. So before we get in any deeper I want to move "
        "the main store over to Postgres."),
    "db_postgres_reason": ("assistant",
        "Agreed, let us go with Postgres. The real deciding factor is the ability to "
        "wrap a set of changes in one transaction. We simply cannot have an invoice "
        "left half written if the process dies in the middle of saving it."),
    "db_replicas": ("user",
        "The invoice list page crawled once we loaded a realistic amount of data. I am "
        "standing up a couple of read replicas and pointing the heavy reporting queries "
        "at them so they stop competing with normal traffic."),
    "cache_add": ("assistant",
        "I put a Redis layer in front of the invoice list, because the same customer "
        "reloads that page over and over. It should take a lot of repeated reads off "
        "the database."),
    "cache_bug": ("user",
        "Something is badly wrong. A customer just emailed to say that for a moment they "
        "could see invoices that clearly belonged to a different company. This is about "
        "as serious as it gets for us."),
    "cache_cause": ("assistant",
        "I found it. The cached invoice list is stored only under the page number, and "
        "nothing about which customer is asking. So two different companies can land on "
        "the very same cached entry. That is the leak."),
    "cache_fix": ("user",
        "The fix is to fold the account into the key we cache under, so every company "
        "gets its own separate entry and can never pick up someone else's page. I am "
        "shipping that now."),
    "auth_decision": ("assistant",
        "For signing in we will hand out short lived web tokens, good for about fifteen "
        "minutes, together with a longer lasting refresh token so that people are not "
        "kicked out every few minutes while they work."),
    "auth_refresh_rotate": ("user",
        "One more thing on sign in: let us replace the refresh token with a fresh one "
        "every time it is redeemed. That way if someone manages to steal one, it is only "
        "good for a single use before it stops working."),
    "deploy_heroku": ("user",
        "Let us just push it to Heroku to begin with. I really do not want to spend our "
        "first week wrestling with servers when we could be building the actual product."),
    "deploy_aws_switch": ("assistant",
        "Heroku has gotten expensive and we keep bumping into its ceilings, so we are "
        "moving where the app runs over to AWS and running it as containers. That gives "
        "us room to grow without the per dyno pricing."),
    "search_consider": ("user",
        "Do we need a dedicated search engine for finding invoices by customer name and "
        "by the notes people leave on them? I keep wondering if we should bring one in."),
    "search_decision": ("assistant",
        "I would really rather not run a separate search cluster yet, it is one more "
        "thing to feed and watch. The full text search built into Postgres is more than "
        "good enough for the amount of data we have, so we will lean on that for now."),
    "billing_job": ("assistant",
        "The overnight job that charges every open invoice is live as of tonight. It "
        "walks the accounts in order and calls the payment provider for each one."),
    "billing_bug": ("user",
        "We charged a handful of customers twice overnight. It looks like the overnight "
        "run went through a second time after something retried it, and nothing stopped "
        "it from repeating the charges."),
    "billing_fix": ("assistant",
        "I gave every charge a unique key that we send to the payment provider, so if the "
        "same charge is attempted again the provider recognizes the key and refuses to "
        "take the money a second time. That closes the double billing."),
    "email_decision": ("user",
        "We will send the invoice emails through SendGrid. Their templates are easy to "
        "work with and the deliverability has been solid in the past, so there is no "
        "reason to build our own mail plumbing."),
    "ratelimit_decision": ("assistant",
        "I set the public interface to allow one hundred requests a minute for each "
        "account. If a serious integration ever needs more we can lift it for them, but "
        "that is a sane default to keep one noisy client from hurting everyone else."),
}

# --------------------------------------------------------------------------
# Distractors: everyday chatter. Several reuse signal entity names on purpose
# so the entity graph has noisy hubs that do not actually answer any question.
# {n} is filled with a running day counter for light, natural variation.
# --------------------------------------------------------------------------
DISTRACTORS: list[tuple[str, str]] = [
    ("user", "Standup, day {n}. I spent most of yesterday on the account settings screen "
             "and I will keep chipping away at it today. Nothing is blocking me right now. "
             "The only real news is that the office coffee machine is making a truly "
             "alarming grinding noise again, and I have a bad feeling it is going to die "
             "in the middle of a Monday when we can least afford to lose it. Someone with "
             "a facilities contact might want to get ahead of that before it is a crisis."),
    ("assistant", "Standup, day {n}. I cleared out a big pile of small review comments and "
                  "bumped a few dependencies to their latest patch versions, nothing scary. "
                  "No blockers on my side. I will be out for a dentist appointment right "
                  "after lunch, so if anything urgent comes up this afternoon please ping me "
                  "on chat rather than expecting me in the room. I should be back online by "
                  "the end of the day to pick up anything that piled up."),
    ("user", "Does anyone want to grab lunch at the Thai place around the corner today? I am "
             "genuinely starving and I could really use twenty minutes away from staring at "
             "the invoice list code, which is starting to blur together. We could also try "
             "the new sandwich spot that opened across from the pharmacy if people are tired "
             "of the usual rotation. Reply here in the next half hour and I will put our "
             "name down for a table so we are not standing around waiting."),
    ("assistant", "I cannot do lunch today, I packed something sad and responsible, but "
                  "please save me a seat for the team dinner on Thursday. Are we still doing "
                  "the place near the station, or did we decide it was too loud last time? "
                  "I remember half the table could not hear the other half. If someone wants "
                  "to pick somewhere quieter I am completely on board, I just want to know "
                  "before I tell my ride what time to grab me."),
    ("user", "Reminder that the all hands got moved to three o'clock on Friday because the "
             "big conference room was double booked by the sales team again. Calendar invites "
             "went out this morning with the updated time, so please let me know if yours did "
             "not refresh. There will be the usual roadmap recap and a short section on "
             "hiring. If you have a topic you want raised during the open questions part, "
             "drop it in the shared doc so we can keep things moving on the day."),
    ("assistant", "The little Postgres elephant sticker on my laptop is peeling off at the "
                  "corner and it is bugging me far more than any reasonable person should be "
                  "bugged by a sticker. Does anyone remember where we got the batch from at "
                  "the last conference? I want a fresh one, and honestly I would grab a "
                  "handful for the new folks too. It is a small thing but a clean laptop lid "
                  "makes the whole day feel slightly more under control."),
    ("user", "The weather looks genuinely grim for the weekend hike, the forecast is nothing "
             "but rain from Friday night straight through Sunday afternoon. I think we should "
             "push it a week rather than slog through the mud and be miserable. I will start "
             "a thread so people can vote on the backup date, and if the following weekend "
             "does not work for enough people we can just aim for later in the month when "
             "things dry out. No sense forcing it and having everyone catch a cold."),
    ("assistant", "Whoever keeps leaving dishes in the sink in the kitchen, this is your "
                  "gentle and only slightly passive aggressive reminder that the dishwasher "
                  "exists and is in fact extremely good at its one job. It takes about ten "
                  "seconds to load a mug. The cleaning crew has started leaving notes, and I "
                  "would rather we sort it out among ourselves than get a stern email from "
                  "building management about kitchen hygiene for the third quarter running."),
    ("user", "I finally cleaned up my browser tabs this morning and discovered three "
             "different tickets I had completely forgotten I was assigned. One of them is "
             "probably stale enough to close outright. Housekeeping is officially done for the "
             "quarter and I refuse to do it again until next year. If anyone else is hoarding "
             "a hundred open tabs like I was, consider this your sign to take ten minutes and "
             "declare bankruptcy on the ones you are never actually going to read."),
    ("assistant", "Did everyone catch the game last night? That ending was something else "
                  "entirely, I was on my feet shouting at the television like it could hear "
                  "me. I lost a small and very friendly bet to my neighbor over it and I am "
                  "still a little annoyed, mostly at myself for betting against the obvious. "
                  "Anyway, if you have no idea what I am talking about, do not worry, normal "
                  "programming resumes immediately and I will stop cluttering the channel."),
    ("user", "The new hire starts Monday and I want their first week to go smoothly. Can "
             "someone volunteer to be their buddy, show them around, and walk them through how "
             "we actually ship things and where the Redis and cache dashboards live so they "
             "are not guessing? It helps enormously to have one named person to ask the dumb "
             "questions to instead of broadcasting every little thing. I will set up the "
             "accounts and laptop ahead of time so day one is not just paperwork and waiting."),
    ("assistant", "Coffee order for the afternoon run, I am heading out in about ten minutes "
                  "and happy to be the mule. I am getting my usual oat milk thing. Reply here "
                  "if you want me to grab you something while I am out, and please be specific "
                  "because last time I guessed on someone's order and got it comprehensively "
                  "wrong. If the line at the good place is out the door again I will fall back "
                  "to the chain on the corner, which is fine but not exciting."),
    ("user", "My personal AWS account sent me a genuinely scary looking bill this month "
             "because I left a toy project running since the spring and forgot about it. This "
             "is completely unrelated to work and I am mostly just venting into the void here, "
             "but let this be your reminder to go check your own side projects before the "
             "month closes. A tiny instance you forgot about will quietly bleed money for "
             "half a year and you will only notice when the statement makes you wince."),
    ("assistant", "Fun fact I learned today that has absolutely nothing to do with anything "
                  "we are building: octopuses have three hearts, and two of them stop beating "
                  "when the animal swims, which is apparently why they would rather crawl. I "
                  "found this out going down a late night reading hole instead of sleeping "
                  "like a responsible adult. That is my entire contribution to the channel "
                  "today, and now I will get back to the actual work, I promise this time."),
    ("user", "The parking garage is closed for resurfacing all of next week, so plan to use "
             "the overflow lot across the street instead. Fair warning that the lot is a "
             "genuine walk, maybe eight or ten minutes, and there is no cover, so bring an "
             "umbrella if the sky looks even slightly threatening. Carpooling might be worth "
             "it for anyone who lives in the same direction. The garage should reopen the "
             "following Monday, assuming the work does not slip, which with these things it "
             "usually does."),
    ("assistant", "I spent twenty minutes updating the team wiki page about the on call "
                  "rotation because it was wildly out of date and still listed two people who "
                  "left the company last year. It now reflects the actual humans who are "
                  "actually reachable. Please have a look when you get a chance and fix your "
                  "own contact details if they are wrong, because the whole thing only works "
                  "if the page is trustworthy at two in the morning when nobody wants to be "
                  "guessing who to wake up."),
    ("user", "Does anyone have a good recommendation for a standing desk that does not cost a "
             "fortune? My back is really not enjoying this sprint and I have reached the point "
             "where I am ready to spend real money to fix it rather than keep complaining. "
             "Bonus points if it is one with a memory setting so I do not have to fiddle with "
             "the height every single morning. If the company has a stipend for this kind of "
             "thing I have completely failed to find the policy, so a pointer there would help "
             "too."),
    ("assistant", "Friendly reminder to submit your expense reports before the end of the "
                  "month, because finance gets understandably grumpy when they trickle in "
                  "late and I genuinely do not blame them for it. It takes five minutes and a "
                  "few photos of receipts. If you lost a receipt there is a form for that, ask "
                  "me and I will send the link. The sooner everyone clears theirs the sooner "
                  "we all get reimbursed, so it is honestly in your own interest to not let it "
                  "sit."),
    ("user", "The search for a conference room microphone that does not make everyone sound "
             "like they are phoning in from inside a tin can continues into its second month. "
             "The current one is a menace to remote attendees. If anyone has used a decent one "
             "at a previous company, please shout, because I am about to start reading reviews "
             "at random and ordering whatever has four stars, which is rarely a winning "
             "strategy. Good audio does more for a meeting than almost anything else and ours "
             "is actively sabotaging us."),
    ("assistant", "I am taking Friday off to visit family out of town, so I will hand off my "
                  "open reviews to whoever is around and has the spare cycles. Nothing is on "
                  "fire on my end, it is all the usual cleanup and a couple of low risk "
                  "changes that can wait until Monday if they need to. I will have my phone if "
                  "something truly breaks, but please treat that as a genuine emergency line "
                  "and not a convenient way to ask me where a config file lives."),
    ("user", "Small appreciation post because it is Friday and we do not do enough of these: "
             "thank you to whoever quietly fixed the flaky test that had been failing on my "
             "machine roughly every third run for two weeks. I was genuinely about to throw "
             "the laptop out of the window, and now the suite is green and calm and I can "
             "trust it again. It is the kind of unglamorous work that nobody notices until it "
             "is done, so I am noticing it loudly here. Drinks are on me next time."),
    ("assistant", "The plants in the corner by the big window are looking pretty rough, a "
                  "couple of them are more brown than green at this point. If you walk past, "
                  "give them a splash of water, because nobody seems to officially own them "
                  "and they are slowly dying of collective neglect. I would adopt them myself "
                  "but I have a documented history of killing even the hardy ones. Maybe we "
                  "start a very low stakes watering rotation, one week each, nothing serious."),
    ("user", "Trivia night at the usual bar is on again this Wednesday and I would like us to "
             "show up in force. Last time we came in a bitter second to the team from "
             "accounting, who were insufferable about it, and I would very much like to "
             "reclaim our honor and the small bar tab that comes with first place. We are "
             "historically strong on geography and weak on pop music, so if anyone here "
             "secretly follows the charts, you are exactly the ringer we need. Starts at "
             "seven, come hungry."),
    ("assistant", "Heads up that building security is testing the fire alarms tomorrow "
                  "morning somewhere between nine and eleven, so please do not panic and "
                  "sprint for the stairs when it goes off, it is scheduled maintenance and not "
                  "a drill or a real emergency for once. They say it will be short bursts "
                  "rather than one long blast. If you are on a customer call, you might want "
                  "to mute preemptively, because the alarm is loud enough to end a conversation "
                  "whether you want it to or not."),
]

# --------------------------------------------------------------------------
# Timeline: the chronological skeleton. ("sig", name) emits a signal;
# ("fill", k) emits k distractors from the pool.
# --------------------------------------------------------------------------
TIMELINE: list[tuple[str, object]] = [
    ("fill", 14),
    ("sig", "db_mongo_plan"), ("sig", "db_mongo_ack"),
    ("fill", 15),
    ("sig", "auth_decision"),
    ("fill", 14),
    ("sig", "deploy_heroku"),
    ("fill", 15),
    ("sig", "cache_add"),
    ("fill", 15),
    ("sig", "email_decision"),
    ("fill", 14),
    ("sig", "billing_job"),
    ("fill", 16),
    ("sig", "search_consider"), ("sig", "search_decision"),
    ("fill", 15),
    ("sig", "ratelimit_decision"),
    ("fill", 17),
    ("sig", "db_postgres_switch"), ("sig", "db_postgres_reason"),
    ("fill", 16),
    ("sig", "cache_bug"), ("sig", "cache_cause"), ("sig", "cache_fix"),
    ("fill", 16),
    ("sig", "billing_bug"), ("sig", "billing_fix"),
    ("fill", 15),
    ("sig", "deploy_aws_switch"),
    ("fill", 16),
    ("sig", "db_replicas"),
    ("fill", 15),
    ("sig", "auth_refresh_rotate"),
    ("fill", 14),
]

# --------------------------------------------------------------------------
# Questions. gold_message_ids are resolved from signal names at build time.
# Reversed-decision questions include the latest decision in the gold set.
# --------------------------------------------------------------------------
QUESTIONS: list[dict] = [
    {"question": "Why did we choose Postgres for the main database?",
     "gold_facts": ["transactions keep an invoice from being half written",
                    "need several rows to change together"],
     "gold": ["db_postgres_switch", "db_postgres_reason"], "vector_friendly": True},
    {"question": "What database did we originally plan to use before we switched?",
     "gold_facts": ["started on MongoDB", "later moved to Postgres"],
     "gold": ["db_mongo_plan", "db_postgres_switch"], "vector_friendly": False},
    {"question": "Why does each company only see its own cached invoice list now?",
     "gold_facts": ["the cache key did not include the account", "account added to the key"],
     "gold": ["cache_cause", "cache_fix"], "vector_friendly": False},
    {"question": "What caused a customer to see invoices that belonged to another company?",
     "gold_facts": ["cache entry keyed only by page", "shared across tenants"],
     "gold": ["cache_bug", "cache_cause"], "vector_friendly": True},
    {"question": "Where are we deploying the app now?",
     "gold_facts": ["moved to AWS", "runs as containers"],
     "gold": ["deploy_aws_switch"], "vector_friendly": True},
    {"question": "What did we use for hosting before we changed it?",
     "gold_facts": ["started on Heroku", "later moved to AWS"],
     "gold": ["deploy_heroku", "deploy_aws_switch"], "vector_friendly": False},
    {"question": "How did we stop the nightly job from charging people twice?",
     "gold_facts": ["idempotency key per charge", "provider rejects a repeat"],
     "gold": ["billing_bug", "billing_fix"], "vector_friendly": False},
    {"question": "Did we decide to use Elasticsearch?",
     "gold_facts": ["chose not to run Elasticsearch", "use Postgres full text search"],
     "gold": ["search_decision"], "vector_friendly": True},
    {"question": "Which service do we use to send invoice emails?",
     "gold_facts": ["SendGrid"],
     "gold": ["email_decision"], "vector_friendly": True},
    {"question": "How long are login tokens valid and how do we keep people signed in?",
     "gold_facts": ["about fifteen minute tokens", "longer refresh token"],
     "gold": ["auth_decision"], "vector_friendly": False},
    {"question": "What change makes a stolen refresh token only usable once?",
     "gold_facts": ["rotate the refresh token on each use"],
     "gold": ["auth_refresh_rotate"], "vector_friendly": False},
    {"question": "Why did the team move away from its first choice of data store?",
     "gold_facts": ["started on MongoDB", "needed transactions", "switched to Postgres"],
     "gold": ["db_mongo_plan", "db_postgres_switch", "db_postgres_reason"],
     "vector_friendly": False},
    {"question": "What did we do about the slow invoice list page?",
     "gold_facts": ["added a Redis cache", "added read replicas for reports"],
     "gold": ["cache_add", "db_replicas"], "vector_friendly": False},
    {"question": "What is the API rate limit?",
     "gold_facts": ["100 requests per minute per account"],
     "gold": ["ratelimit_decision"], "vector_friendly": True},
]


def build():
    messages: list[dict] = []
    name_to_id: dict[str, str] = {}
    fill_index = 0
    day = 0

    def emit(role: str, content: str) -> str:
        node_id = f"m{len(messages)}"
        messages.append({"id": node_id, "role": role, "content": content})
        return node_id

    def emit_distractor():
        nonlocal fill_index, day
        role, template = DISTRACTORS[fill_index % len(DISTRACTORS)]
        fill_index += 1
        day += 1
        emit(role, template.format(n=day) if "{n}" in template else template)

    for kind, value in TIMELINE:
        if kind == "sig":
            role, content = SIGNALS[value]
            name_to_id[value] = emit(role, content)
        else:
            for _ in range(int(value)):
                emit_distractor()

    counter = DefaultTokenCounter()
    total = sum(counter.count(m["content"]) for m in messages)
    assert total >= 20000, f"conversation is only {total} tokens, need at least 20000"

    def total_tokens() -> int:
        return total

    questions = []
    for q in QUESTIONS:
        questions.append({
            "question": q["question"],
            "gold_facts": q["gold_facts"],
            "gold_message_ids": [name_to_id[name] for name in q["gold"]],
            "gold_signal_names": q["gold"],
            "vector_friendly": q["vector_friendly"],
        })

    (HERE / "conversation.json").write_text(json.dumps(messages, indent=2), encoding="utf-8")
    (HERE / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")

    print(f"messages: {len(messages)}")
    print(f"tokens (cl100k_base, approximate={counter.approximate}): {total_tokens()}")
    print(f"questions: {len(questions)}")
    print(f"vector_friendly questions: {sum(q['vector_friendly'] for q in QUESTIONS)}")


if __name__ == "__main__":
    build()
