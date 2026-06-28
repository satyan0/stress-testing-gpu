"""
Tender-executive prompts for realistic profiling.

Instead of generic filler, each prompt puts the model in the seat of a senior
tender / bid executive, hands it an actual NIT (tender notice) as context, and
asks a real bid-desk question (eligibility, EMD, scope, risk clauses, bid/no-bid).

The prompt-size sweep is preserved: the tender document is grown to each target
token length by appending more GCC/SCC clauses, then the question is appended
intact at the very end. Output is rendered through the model's chat template so
the model actually adopts the persona.

build_prompt(tokenizer, n_tokens, q_idx) -> str   (chat-rendered)
build_prompts(tokenizer, sizes)         -> {size: str}   (same signature as before)
"""

SYSTEM = (
    "You are a senior tender executive at an engineering contracting firm. "
    "You read government tender notices (NITs) and advise the bid desk. "
    "Answer precisely and cite the specific clause or field you rely on. "
    "If information is missing, say so rather than assuming. Be concise and "
    "decision-oriented."
)

TENDER_HEADER = """\
TENDER NOTICE (NIT)
NIT No.: CE/EW/2026-27/0418
Name of Work: Design, supply, installation, testing and commissioning of a
  33/11 kV electrical substation including allied civil works.
Procuring Entity: State Power Transmission Corporation Ltd.
Tender Value (estimated): INR 18.42 Crore
Earnest Money Deposit (EMD): INR 18,42,000 (1% of tender value), via BG or
  online payment on the GeM/CPPP portal; valid 180 days.
Tender Fee: INR 11,800 (non-refundable), online only.
Bid Type: Two-cover (Technical + Financial), e-procurement.
Completion Period: 14 months from date of LOA.
Defect Liability Period: 24 months from commissioning.
"""

ELIGIBILITY = """\
ELIGIBILITY CRITERIA
E1. Average annual turnover in the last 3 financial years >= INR 9.00 Crore.
E2. Similar work experience: one work of >= INR 14.7 Cr, OR two of >= INR 9.2 Cr,
    OR three of >= INR 7.4 Cr, executed in the last 7 years.
E3. "Similar work" = EPC of substations of 33 kV or above for a govt/PSU client.
E4. Valid electrical contractor licence (HT) issued by the State Licensing Board.
E5. Positive net worth in each of the last 3 financial years.
E6. No work abandoned/terminated for default in the last 5 years (self-declared).
E7. Registration on GeM and the State e-procurement portal; valid GST and PAN.
"""

SCOPE = """\
SCOPE OF WORK
Design and detailed engineering of the 33/11 kV substation; supply of 2 x 10 MVA
power transformers, 33 kV and 11 kV switchgear, control & relay panels, battery
and charger, earthing and lightning protection; associated civil works (control
room, transformer plinths, cable trenches, boundary wall); SCADA-ready RTU
integration; testing, commissioning and 24-month O&M support.
"""

BOQ = """\
BILL OF QUANTITIES (extract)
B1. 10 MVA 33/11 kV power transformer .......... 2 nos.
B2. 33 kV VCB outdoor .......................... 4 nos.
B3. 11 kV indoor switchgear panel .............. 8 nos.
B4. Control & relay panel ...................... 6 nos.
B5. Substation earthing (complete) ............. 1 lot
B6. Control room civil works ................... 1 lot
"""

# Pool of realistic GCC/SCC clauses used to grow the document to a target size.
CLAUSE_POOL = [
    "Price basis shall be firm and fixed; no price variation/escalation is "
    "admissible for the entire contract period including authorised extensions.",
    "Liquidated damages of 0.5% of contract value per week of delay shall apply, "
    "subject to a ceiling of 10% of the contract value.",
    "A performance security of 5% of the contract value shall be furnished as an "
    "unconditional bank guarantee within 14 days of issue of the LOA.",
    "10% of each running account bill shall be withheld as retention money, "
    "released against the defect liability bank guarantee at completion.",
    "All taxes and duties shall be deemed included in the quoted rates except GST, "
    "which shall be reimbursed at actuals against valid tax invoices.",
    "The contractor shall comply with all applicable labour laws, EPF/ESI "
    "registration, and submit monthly compliance certificates.",
    "Materials shall conform to the relevant IS/IEC standards; type test "
    "certificates not older than 5 years shall be submitted for switchgear.",
    "The employer reserves the right to vary quantities by +/-25% at the same "
    "rates without any compensation to the contractor.",
    "Disputes shall be referred to arbitration under the Arbitration and "
    "Conciliation Act, 1996; the seat of arbitration shall be the state capital.",
    "The bidder shall submit a detailed PERT/CPM programme within 10 days of LOA; "
    "monthly physical and financial progress reports are mandatory.",
    "Insurance (CAR policy, third-party, and workmen's compensation) shall be "
    "taken by the contractor in the joint names of employer and contractor.",
    "Make of major equipment shall be from the employer-approved vendor list; "
    "any substitution requires prior written approval of the Engineer.",
    "Mobilisation advance up to 5% may be released against an equal bank "
    "guarantee, recoverable from running bills with interest at SBI MCLR + 2%.",
    "The technical bid shall be opened first; only technically responsive bidders "
    "shall have their financial bids opened on a date intimated separately.",
    "Conditional bids, or bids with deviations to the commercial terms, are liable "
    "to be summarily rejected.",
    "The contractor shall provide a 24-month comprehensive O&M including spares, "
    "with a guaranteed response time of 4 hours for critical faults.",
    "Site possession shall be given in phases; the contractor shall plan works to "
    "suit phased handover without any idle-charges claim.",
    "Safety: the contractor shall deploy a qualified safety officer; any fatal "
    "accident attracts a penalty and possible debarment.",
    "Bid validity shall be 180 days from the date of bid opening; EMD shall remain "
    "valid for 45 days beyond bid validity.",
    "Evaluation: L1 shall be determined on the overall quoted price inclusive of "
    "all items; item-wise reasonableness may be sought before award.",
]

QUESTIONS = [
    "Based on the NIT, are we eligible to bid? Check each eligibility criterion "
    "and state clearly which ones we can meet and which need verification.",
    "What is the EMD amount, the acceptable modes of submission, and its required "
    "validity? Cite the exact figures and clause.",
    "Summarise the scope of work and flag the two or three most technically "
    "demanding deliverables we should resource carefully.",
    "Identify the most onerous commercial clauses (LD, retention, price basis, "
    "performance security) and tell the bid desk what to price in for risk.",
    "Give a bid / no-bid recommendation with a short rationale, assuming our "
    "average turnover is INR 11 Cr and our largest similar work was INR 12.5 Cr.",
    "List the documents that must accompany the technical bid for it to be "
    "treated as responsive.",
]


def _grow_context(tokenizer, n_tokens: int) -> str:
    """Build the tender document body sized to ~n_tokens tokens."""
    body = "\n".join([TENDER_HEADER, ELIGIBILITY, SCOPE, BOQ,
                      "GENERAL & SPECIAL CONDITIONS OF CONTRACT"])
    i = 0
    # Append renumbered clauses until we reach the target token budget.
    while len(tokenizer(body, add_special_tokens=False)["input_ids"]) < n_tokens:
        clause = CLAUSE_POOL[i % len(CLAUSE_POOL)]
        body += f"\nC{i + 1}. {clause}"
        i += 1
    # Trim to the exact target (may cut mid-clause -- fine, the question follows).
    ids = tokenizer(body, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids)


def build_prompt(tokenizer, n_tokens: int, q_idx: int = 0) -> str:
    context = _grow_context(tokenizer, n_tokens)
    question = QUESTIONS[q_idx % len(QUESTIONS)]
    user = (
        "Here is the tender notice to review:\n\n"
        f"{context}\n\n"
        f"QUESTION: {question}"
    )
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for tokenizers without a chat template.
        return f"{SYSTEM}\n\n{user}\n\nAnswer:"


def build_prompts(tokenizer, sizes):
    # Rotate the question across sizes so each run exercises a different task.
    return {n: build_prompt(tokenizer, n, q_idx=i) for i, n in enumerate(sizes)}