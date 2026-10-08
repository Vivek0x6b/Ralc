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
# Distractors: everyday chatter, generated so that every message is unique.
# Many of them reuse the same entity names as the signals (cache, Postgres,
# AWS, billing, search, SendGrid, Redis, invoices, rate limit) in off-topic
# ways, so the entity graph has realistic noisy hubs that answer no question.
# --------------------------------------------------------------------------
_NAMES = ["Priya", "Sam", "Diego", "Mei", "Jordan", "Aisha", "Tom", "Lena",
          "Omar", "Riya", "Chen", "Noah", "Kira", "Raj", "Sofia", "Will",
          "Hana", "Marcus", "Ivy", "Dana"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_FOODS = ["ramen", "tacos", "pho", "burritos", "dumplings", "falafel", "pizza",
          "curry", "sushi", "bibimbap", "pad thai", "sandwiches", "poke", "gyros"]
_PLACES = ["the Thai place", "the new deli", "the food truck by the park",
           "the ramen counter", "the cafe on Fifth", "the burger joint",
           "the salad spot", "the taqueria", "the noodle bar", "the bakery",
           "the diner", "the dumpling house"]
_HOBBIES = ["bouldering", "cycling", "pottery", "board games", "running",
            "baking", "chess", "photography", "hiking", "yoga", "painting",
            "gardening", "birdwatching", "climbing"]
_TEAMS = ["accounting", "the design team", "marketing", "the sales crew",
          "the ops team", "the data team", "the interns", "the support desk",
          "the mobile team", "the platform team"]
_ENTITIES = [
    "the meeting room someone keeps renaming to the cache",
    "my personal AWS bill from a forgotten side project",
    "the Postgres elephant sticker everyone has but me",
    "the billing team two floors down who hog the good room",
    "an Elasticsearch meetup happening downtown",
    "the SendGrid newsletter I subscribed to by accident years ago",
    "a tiny Redis plushie someone left on the printer",
    "the metaphorical pile of invoices in my inbox",
    "the office wifi that seems to have its own rate limit",
    "my endless search for a parking spot",
    "the read replica of my to-do list that never syncs",
    "the Heroku hoodie I won and never wear",
]

_ASIDES = [
    "On a separate note, the coffee machine situation remains dire and I have developed opinions about it that nobody asked me to share.",
    "Unrelated, but if someone finds a blue pen wandering the office, it is mine and it has survived three laptops and counting.",
    "Side thought, the elevator music has looped the same tune all week and it is quietly rewriting the inside of my skull.",
    "Separately, whoever disguised the wifi password as a cryptic riddle, I respect the artistry and resent the daily inconvenience equally.",
    "In other news, my plant at home is thriving while the office ones wilt, which surely says something I am not ready to hear.",
    "Tangent, the new chairs manage to be worse than the old chairs, a genuine achievement I did not believe was physically possible.",
    "Totally aside, I keep misreading the renamed meeting room as the cache and marching confidently to entirely the wrong floor.",
    "Off topic, my phone autocorrects Postgres to postgrad every single time and at this point I have simply surrendered to it.",
    "Separately, the parking lot remains a daily negotiation with fate and one extremely territorial pigeon who was clearly there first.",
    "On another note, my personal AWS bill and I are no longer on speaking terms after last month's quiet and expensive betrayal.",
    "Unrelated, the office plants have unionized in spirit if not in fact, and frankly their list of demands seems pretty reasonable.",
    "Side note, I finally labeled my lunch in the fridge, which feels like the most responsible adult thing I have done all quarter.",
    "Aside, the SendGrid newsletter I subscribed to by accident years ago still arrives weekly and I have grown oddly fond of it.",
    "Separately, someone keeps borrowing the good stapler and returning it slightly worse, which is somehow more sinister than just keeping it.",
    "In passing, the stairwell smells aggressively of someone's lunch and I have theories but no proof and no real desire for either.",
    "Tangent, my search for a decent desk lamp has entered its third week and I am starting to seriously question my own standards.",
    "Off topic, the Redis plushie on the printer acquired a tiny hat overnight and nobody is claiming responsibility for the upgrade.",
    "Separately, I rate the new hand soap a solid seven, docking points only for the aggressively artificial green apple situation.",
    "On a side note, the billing team downstairs got a popcorn machine and the smell drifts up here like a taunt every afternoon.",
    "Unrelated, I have started taking the long way to the kitchen just to hit my step count, and I already regret telling you this.",
    "Aside, the whiteboard markers are all mysteriously dead again, a recurring tragedy that honestly deserves its own small memorial.",
    "Separately, my calendar and I disagree about how many meetings a human can attend before becoming a ghost, and it is winning.",
    "In other news, the vending machine now accepts taps, a leap forward that has dangerously lowered the barrier to buying chips.",
    "Off topic, I found a sticky note from my first week that just says be brave, and I have no memory of why, but I am keeping it.",
]

_TEMPLATES: list[tuple[str, str]] = [
    ("user", "Sprint {wk} check in: I pushed on the account settings screen and I am not "
             "blocked. Unrelated, but {entity} has been living in my head today and quietly "
             "derailing me, so if you catch me staring into the middle distance, that is why."),
    ("assistant", "Sprint {wk} note: reviews cleared, a couple of dependencies bumped, no "
                  "blockers. I am out after lunch, {name} kindly agreed to cover anything "
                  "urgent, so route the fires their way and the rest can wait until I am back."),
    ("user", "Anyone want {food} at {place} today? I badly need to step away from the screen, "
             "and {name} is in if we leave before the {day} rush turns the place into a zoo."),
    ("assistant", "Cannot do lunch, but save me a seat at the team dinner on {day}. Last time "
                  "{place} was so loud that {name} and I gave up and mimed across the table, "
                  "which was its own kind of fun but genuinely not how I want to catch up."),
    ("user", "Heads up, the all hands slid to {day} afternoon because the big room got double "
             "booked by {team} again. Invites are updated, so ping {name} if yours did not move."),
    ("assistant", "Does anyone remember where we got {entity}? I would like one, and I would "
                  "happily grab a few extras for the people starting next {day} so they feel "
                  "properly welcomed and not like they wandered in off the street by mistake."),
    ("user", "The weekend forecast looks rough, so {name} and I want to push the {hobby} plan "
             "to the following {day}. I will open a thread to pick a backup, no pressure at all."),
    ("assistant", "Kitchen reminder: the dishwasher exists and is excellent at its one job. "
                  "{name} is innocent this round, but the sink has become a modern art piece "
                  "and the cleaning crew has started leaving notes that are getting pointed."),
    ("user", "Cleaned up my browser tabs and found three forgotten tickets, plus a tab about "
             "{entity} that explains a surprising amount about how my whole week has gone."),
    ("assistant", "Did anyone catch the match last night? {team} fell apart at the very end, "
                  "and now {name} owes me a {food} lunch, which I fully intend to collect on {day}."),
    ("user", "The new hire starts {day}. Can someone be their buddy, show them the ropes, where "
             "the dashboards live, and how we ship? {name} set a high bar doing it last time."),
    ("assistant", "Coffee run in ten minutes, heading past {place}. {name} wants the usual. "
                  "Shout if you want anything, and please be specific, because my guesses have a "
                  "genuinely terrible track record and I refuse to take the blame again."),
    ("user", "Please file your expenses before month end. {name} in finance gets rightfully "
             "grumpy when they trickle in late, and honestly I would be grumpy too in their shoes."),
    ("assistant", "My back has filed a formal complaint about this chair. Anyone have a standing "
                  "desk they love? {name} swears by theirs, though it costs about as much as a "
                  "serious {hobby} habit, which is either a warning or a dare depending on the day."),
    ("user", "Trivia at the usual bar on {day}. We lost to {team} last time and I want our honor "
             "back. We are strong on geography and weak on pop music, so ringers are very welcome."),
    ("assistant", "Security is testing the fire alarms on {day} morning, so do not sprint for the "
                  "exits when it blares. {name} got startled last year and sent a coffee clean "
                  "across standup, an incident we lovingly bring up at every possible opportunity."),
    ("user", "The garage is closed next week, use the lot across the street. It is a walk, bring "
             "an umbrella, and maybe carpool with {name} if you are coming from the same direction."),
    ("assistant", "Appreciation for {name}, who quietly killed the flaky test that haunted me for "
                  "a {hobby}-length stretch of afternoons. The suite is calm again and so am I, "
                  "finally. Drinks on me on {day}, and that is a firm promise, not a vague gesture."),
    ("user", "The window plants are dying of neglect, so {name} and I are starting a very low "
             "stakes watering rotation, one {day} each. Join us, the bar for entry is on the floor."),
    ("assistant", "Fell down a reading hole last night instead of sleeping and learned that sea "
                  "otters hold hands so they do not drift apart. {name}, this is your daily dose of "
                  "nothing to do with work, and now back to the grind I reluctantly return."),
    ("user", "{name} is organizing a {hobby} outing on {day} for anyone keen. No pressure, but it "
             "has been a long stretch and a change of scenery somewhere past {place} would do us good."),
    ("assistant", "The conference room mic still makes everyone sound like a drive through order. "
                  "{name} is hunting for a replacement, recommendations welcome, our remote folks "
                  "have suffered long enough and are beginning to take the audio quality personally."),
    ("user", "Quick poll: {food} or {food2} for the team lunch on {day}? {name} cannot eat one of "
             "them, so if it is close let us be kind and quietly pick the other one without a debate."),
    ("assistant", "Taking {day} off to see family. {name} will cover my reviews. Nothing is on fire, "
                  "just routine cleanup, and I will keep my phone on for genuine emergencies only, "
                  "which firmly does not include asking me where a particular config file lives."),
    ("user", "Someone left {entity} near the kitchen and it has been sitting there for days now. "
             "If it is yours, please rescue it before the cleaning crew stages a quiet intervention."),
    ("assistant", "The coffee machine is making the alarming noise again. {name} gives it maybe a "
                  "{day} before it quits for good. Can facilities look before it dies mid morning and "
                  "we are all forced to be our unfiltered, uncaffeinated selves in the standup?"),
    ("user", "Shout out to {name} for a great onboarding session. We hit the basics and somehow "
             "ended up deep in {entity}, which the new folks found weirdly fascinating, bless them."),
    ("assistant", "The {hobby} club meets {day} after work and {name} is bringing spare gear for "
                  "anyone who wants to try before buying. Low commitment, high fun, mild bruising."),
    ("user", "Fixed my home setup after wrestling with it all {day}. Totally unrelated to work, but "
             "it reminded me how rare and lovely it is when a thing just works on the very first try."),
    ("assistant", "The vending machine ate {name}'s money again and I witnessed the whole quiet "
                  "tragedy unfold. Facilities has been emailed. I will not rest until justice, or at "
                  "the very least a refunded bag of pretzels, is finally served to the people."),
    ("user", "Weekend recap: I tried {hobby} for the first time and it went about as well as you "
             "would expect. Ask {name} for the deeply unflattering photos. Would humiliate myself again."),
    ("assistant", "Can we stop booking meetings over lunch on {day}? {name} and I keep choosing "
                  "between {food} and being informed, and that is a genuinely cruel thing to do to people."),
    ("user", "The elevator is out until {day}, so it is stairs only for now. {name} says we are all "
             "getting our steps in, which is the kind of relentless optimism I admire and find exhausting."),
    ("assistant", "Started a book club, the first pick is short so nobody panics. {name} is in, we "
                  "meet {day} at the cafe near {place}. Newcomers very welcome, strong opinions optional."),
    ("user", "Anyone know a good dentist nearby? Mine retired and {name} recommended theirs, but it "
             "is a {hobby}-sized trek across town and I am not convinced my molars are worth the commute."),
    ("assistant", "Parking tip from {name}, discovered entirely by accident: the lot near {place} is "
                  "free after six. Filing that away for the next late crunch, not that we would ever plan one."),
]


def _distractor(i: int) -> tuple[str, str]:
    role, template = _TEMPLATES[i % len(_TEMPLATES)]
    fills = {
        "name": _NAMES[(i * 3 + 1) % len(_NAMES)],
        "day": _DAYS[(i * 2) % len(_DAYS)],
        "food": _FOODS[(i * 5) % len(_FOODS)],
        "food2": _FOODS[(i * 5 + 3) % len(_FOODS)],
        "place": _PLACES[(i * 4 + 2) % len(_PLACES)],
        "hobby": _HOBBIES[(i * 6 + 1) % len(_HOBBIES)],
        "team": _TEAMS[(i * 7 + 3) % len(_TEAMS)],
        "entity": _ENTITIES[(i * 3 + 2) % len(_ENTITIES)],
        "wk": i,
    }
    body = template.format(**fills)
    aside1 = _ASIDES[(i * 11 + 5) % len(_ASIDES)]
    aside2 = _ASIDES[(i * 7 + 2) % len(_ASIDES)]
    if aside1 == aside2:
        aside2 = _ASIDES[(i * 7 + 3) % len(_ASIDES)]
    return role, f"{body} {aside1} {aside2}"


# --------------------------------------------------------------------------
# Timeline: the chronological skeleton. ("sig", name) emits a signal;
# ("fill", k) emits k distractors from the pool.
# --------------------------------------------------------------------------
TIMELINE: list[tuple[str, object]] = [
    ("fill", 10),
    ("sig", "db_mongo_plan"), ("sig", "db_mongo_ack"),
    ("fill", 10),
    ("sig", "auth_decision"),
    ("fill", 10),
    ("sig", "deploy_heroku"),
    ("fill", 10),
    ("sig", "cache_add"),
    ("fill", 10),
    ("sig", "email_decision"),
    ("fill", 10),
    ("sig", "billing_job"),
    ("fill", 10),
    ("sig", "search_consider"), ("sig", "search_decision"),
    ("fill", 10),
    ("sig", "ratelimit_decision"),
    ("fill", 10),
    ("sig", "db_postgres_switch"), ("sig", "db_postgres_reason"),
    ("fill", 10),
    ("sig", "cache_bug"), ("sig", "cache_cause"), ("sig", "cache_fix"),
    ("fill", 10),
    ("sig", "billing_bug"), ("sig", "billing_fix"),
    ("fill", 10),
    ("sig", "deploy_aws_switch"),
    ("fill", 10),
    ("sig", "db_replicas"),
    ("fill", 10),
    ("sig", "auth_refresh_rotate"),
    ("fill", 10),
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
    seen_contents: set[str] = set()
    fill_index = 0

    def emit(role: str, content: str) -> str:
        node_id = f"m{len(messages)}"
        messages.append({"id": node_id, "role": role, "content": content})
        return node_id

    def emit_distractor():
        nonlocal fill_index
        while True:
            role, content = _distractor(fill_index)
            fill_index += 1
            if content not in seen_contents:
                seen_contents.add(content)
                emit(role, content)
                return

    for kind, value in TIMELINE:
        if kind == "sig":
            role, content = SIGNALS[value]
            name_to_id[value] = emit(role, content)
        else:
            for _ in range(int(value)):
                emit_distractor()

    counter = DefaultTokenCounter()
    total = sum(counter.count(m["content"]) for m in messages)
    # Top up with unique distractors (appended after all signals, so the gold
    # ids stay stable) until the conversation clears the token target.
    while total < TOKEN_TARGET:
        emit_distractor()
        total += counter.count(messages[-1]["content"])

    unique = len({m["content"] for m in messages})
    assert unique == len(messages), f"messages are not all unique: {unique}/{len(messages)}"
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
    print(f"unique messages: {len({m['content'] for m in messages})}")
    print(f"tokens (cl100k_base, approximate={counter.approximate}): {total_tokens()}")
    print(f"questions: {len(questions)}")
    print(f"vector_friendly questions: {sum(q['vector_friendly'] for q in QUESTIONS)}")


if __name__ == "__main__":
    build()
