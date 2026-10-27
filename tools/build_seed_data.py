"""Build the seed finance and tax examples with answers computed in code.

Every answer below is calculated, not typed by hand, so the worked reasoning
and the final answer always agree. Run from the kit folder:

    python tools/build_seed_data.py

It writes data/train.jsonl, data/eval.jsonl and data/seed_examples.csv.
All rates and rules a question needs are stated inside the question, so the
answers do not depend on tax law that may change.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def money(x: float) -> str:
    return f"${x:,.2f}"


def npv(rate: float, cashflows: list[float]) -> float:
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def pmt(principal: float, annual_rate: float, years: int, per_year: int = 12) -> float:
    r = annual_rate / per_year
    n = years * per_year
    return principal * r / (1 - (1 + r) ** -n)


def bond_price(face: float, coupon_rate: float, ytm: float, years: int) -> float:
    c = face * coupon_rate
    return sum(c / (1 + ytm) ** t for t in range(1, years + 1)) + face / (1 + ytm) ** years


# Each builder returns: (question, reasoning, answer, answer_type, tolerance)
# answer_type: "number" (compare within relative tolerance), "percent" (a percent
# value such as 22.47 meaning 22.47%), or "choice" (a single letter).

def corp_tax(pretax, fed, state):
    st = pretax * state
    fd = (pretax - st) * fed
    total = st + fd
    q = (f"A company has pre-tax income of {money(pretax)}. The state tax rate is {state:.0%} of pre-tax income, "
         f"and state tax is deductible when computing federal taxable income. The federal rate is {fed:.0%}. "
         f"What is the total income tax (state plus federal), in dollars?")
    r = (f"State tax = {money(pretax)} × {state:.0%} = {money(st)}.\n"
         f"Federal taxable income = {money(pretax)} − {money(st)} = {money(pretax - st)}.\n"
         f"Federal tax = {money(pretax - st)} × {fed:.0%} = {money(fd)}.\n"
         f"Total = {money(st)} + {money(fd)} = {money(total)}.")
    return q, r, round(total, 2), "number", 0.001


def effective_rate(expense, pretax):
    rate = expense / pretax * 100
    q = f"Income tax expense is {money(expense)} on pre-tax income of {money(pretax)}. What is the effective tax rate, in percent?"
    r = f"Effective tax rate = {money(expense)} ÷ {money(pretax)} = {rate:.2f}%."
    return q, r, round(rate, 2), "percent", 0.001


def straight_line_book_value(cost, salvage, life, year):
    annual = (cost - salvage) / life
    bv = cost - annual * year
    q = (f"Equipment costs {money(cost)}, has a salvage value of {money(salvage)} and a useful life of {life} years. "
         f"Using straight-line depreciation, what is its book value at the end of year {year}?")
    r = (f"Annual depreciation = ({money(cost)} − {money(salvage)}) ÷ {life} = {money(annual)}.\n"
         f"Accumulated depreciation after {year} years = {money(annual * year)}.\n"
         f"Book value = {money(cost)} − {money(annual * year)} = {money(bv)}.")
    return q, r, round(bv, 2), "number", 0.001


def after_tax_cost_of_debt(kd, t):
    v = kd * (1 - t) * 100
    q = f"A company's pre-tax cost of debt is {kd:.1%} and its tax rate is {t:.0%}. What is its after-tax cost of debt, in percent?"
    r = f"After-tax cost of debt = {kd:.1%} × (1 − {t:.0%}) = {v:.2f}%."
    return q, r, round(v, 2), "percent", 0.002


def cagr(start, end, years):
    v = ((end / start) ** (1 / years) - 1) * 100
    q = f"Revenue grew from {money(start)} to {money(end)} over {years} years. What is the compound annual growth rate (CAGR), in percent?"
    r = f"CAGR = ({money(end)} ÷ {money(start)})^(1/{years}) − 1 = {v:.2f}%."
    return q, r, round(v, 2), "percent", 0.002


def loan_payment(principal, rate, years):
    p = pmt(principal, rate, years)
    q = (f"A {money(principal)} loan has a {rate:.1%} annual interest rate, compounded monthly, and is repaid in equal "
         f"monthly payments over {years} years. What is the monthly payment?")
    r = (f"Monthly rate r = {rate:.1%} ÷ 12 = {rate / 12:.6f}; number of payments n = {years * 12}.\n"
         f"Payment = P × r ÷ (1 − (1 + r)^−n) = {money(p)}.")
    return q, r, round(p, 2), "number", 0.001


def project_npv(rate, outlay, inflow, years):
    cfs = [-outlay] + [inflow] * years
    v = npv(rate, cfs)
    q = (f"A project costs {money(outlay)} today and returns {money(inflow)} at the end of each year for {years} years. "
         f"At a {rate:.0%} discount rate, what is its net present value (NPV)?")
    r = (f"Present value of inflows = {money(inflow)} × annuity factor at {rate:.0%} for {years} years = {money(v + outlay)}.\n"
         f"NPV = {money(v + outlay)} − {money(outlay)} = {money(v)}.")
    return q, r, round(v, 2), "number", 0.001


def post_money_safe(amount, cap):
    v = amount / cap * 100
    q = (f"An investor buys a {money(amount)} post-money SAFE (simple agreement for future equity) with a "
         f"{money(cap)} post-money valuation cap. Ignoring any later dilution, what percentage of the company "
         f"does the investor own when the SAFE converts at the cap?")
    r = f"For a post-money SAFE, ownership = investment ÷ post-money cap = {money(amount)} ÷ {money(cap)} = {v:.2f}%."
    return q, r, round(v, 2), "percent", 0.001


def convertible_note(principal, rate, months, discount, cap, pre_shares, round_price):
    accrued = principal * (1 + rate * months / 12)
    cap_price = cap / pre_shares
    disc_price = round_price * (1 - discount)
    conv = min(cap_price, disc_price)
    shares = accrued / conv
    q = (f"A {money(principal)} convertible note accrues {rate:.0%} simple annual interest for {months} months, then "
         f"converts in a priced round at {money(round_price)} per share. The note has a {discount:.0%} discount and a "
         f"{money(cap)} pre-money valuation cap, and the company has {pre_shares:,} fully diluted shares before the round. "
         f"The note converts at the lower of the cap price and the discount price. How many shares does the noteholder receive?")
    r = (f"Principal plus interest = {money(principal)} × (1 + {rate:.0%} × {months}/12) = {money(accrued)}.\n"
         f"Cap price = {money(cap)} ÷ {pre_shares:,} = {money(cap_price)}. Discount price = {money(round_price)} × (1 − {discount:.0%}) = {money(disc_price)}.\n"
         f"Conversion price = lower of the two = {money(conv)}.\n"
         f"Shares = {money(accrued)} ÷ {money(conv)} = {shares:,.0f}.")
    return q, r, round(shares, 0), "number", 0.001


def dilution(founder, total, new):
    v = founder / (total + new) * 100
    q = (f"A founder holds {founder:,} of {total:,} fully diluted shares. A new financing issues {new:,} new shares. "
         f"What is the founder's ownership after the round, in percent?")
    r = f"Ownership = {founder:,} ÷ ({total:,} + {new:,}) = {v:.2f}%."
    return q, r, round(v, 2), "percent", 0.001


def option_pool_shuffle(pre, invest, pool, existing):
    post = pre + invest
    inv_pct = invest / post
    existing_pct = 1 - inv_pct - pool
    total_post = existing / existing_pct
    price = post / total_post
    q = (f"A startup raises {money(invest)} at a {money(pre)} pre-money valuation. The term sheet requires an unissued option "
         f"pool equal to {pool:.0%} of the post-money fully diluted shares, created before the investment. Existing holders "
         f"own {existing:,} shares and there is no pool today. What is the price per share in the round?")
    r = (f"Post-money = {money(pre)} + {money(invest)} = {money(post)}. Investor owns {inv_pct:.0%}; the new pool is {pool:.0%}; "
         f"existing holders keep {existing_pct:.0%}.\n"
         f"Post-money fully diluted shares = {existing:,} ÷ {existing_pct:.0%} = {total_post:,.2f}.\n"
         f"Price per share = {money(post)} ÷ {total_post:,.2f} = ${price:.4f}.")
    return q, r, round(price, 4), "number", 0.001


def current_ratio(ca, cl):
    v = ca / cl
    q = f"Current assets are {money(ca)} and current liabilities are {money(cl)}. What is the current ratio?"
    r = f"Current ratio = {money(ca)} ÷ {money(cl)} = {v:.2f}."
    return q, r, round(v, 2), "number", 0.005


def ebitda(ni, interest, taxes, da):
    v = ni + interest + taxes + da
    q = (f"Net income is {money(ni)}, interest expense is {money(interest)}, income tax expense is {money(taxes)}, and "
         f"depreciation and amortization total {money(da)}. What is EBITDA (earnings before interest, taxes, depreciation and amortization)?")
    r = f"EBITDA = {money(ni)} + {money(interest)} + {money(taxes)} + {money(da)} = {money(v)}."
    return q, r, round(v, 2), "number", 0.001


def cap_gains_tax(basis, proceeds, ltcg, niit):
    gain = proceeds - basis
    tax = gain * (ltcg + niit)
    q = (f"An individual sells stock for {money(proceeds)} that was bought for {money(basis)} more than one year earlier. "
         f"Assume a {ltcg:.0%} long-term capital gains rate plus a {niit:.1%} net investment income tax on the full gain. "
         f"What is the total federal tax on the gain?")
    r = (f"Gain = {money(proceeds)} − {money(basis)} = {money(gain)}.\n"
         f"Tax = {money(gain)} × ({ltcg:.0%} + {niit:.1%}) = {money(tax)}.")
    return q, r, round(tax, 2), "number", 0.001


def tvpi(paid_in, dist, nav):
    v = (dist + nav) / paid_in
    q = (f"A venture fund has called {money(paid_in)} from its limited partners, distributed {money(dist)}, and holds "
         f"remaining investments valued at {money(nav)}. What is its TVPI (total value to paid-in capital) multiple?")
    r = f"TVPI = ({money(dist)} + {money(nav)}) ÷ {money(paid_in)} = {v:.2f}x."
    return q, r, round(v, 2), "number", 0.005


def european_waterfall(capital, pref, years, dist, carry):
    pref_amt = capital * pref * years
    remaining = dist - capital - pref_amt
    catchup = carry / (1 - carry) * pref_amt          # GP catches up to carry% of profits
    catchup = min(catchup, max(remaining, 0))
    rest = remaining - catchup
    gp = catchup + rest * carry
    q = (f"A fund's limited partners contributed {money(capital)}. The fund has distributed {money(dist)} in total. "
         f"The waterfall is: (1) return of contributed capital; (2) an {pref:.0%} simple annual preferred return for {years} years; "
         f"(3) a 100% general partner catch-up until the general partner has received {carry:.0%} of all profits; "
         f"(4) the remainder split {1 - carry:.0%}/{carry:.0%} between limited and general partners. "
         f"How much carried interest does the general partner receive?")
    r = (f"Return of capital: {money(capital)}. Preferred return: {money(capital)} × {pref:.0%} × {years} = {money(pref_amt)}.\n"
         f"Remaining after steps 1–2: {money(dist)} − {money(capital)} − {money(pref_amt)} = {money(remaining)}.\n"
         f"Catch-up: the general partner needs {carry:.0%}/{1 - carry:.0%} of the preferred return = {money(catchup)}.\n"
         f"Remainder after catch-up: {money(rest)}; general partner share {carry:.0%} = {money(rest * carry)}.\n"
         f"Total carried interest = {money(catchup)} + {money(rest * carry)} = {money(gp)}.")
    return q, r, round(gp, 2), "number", 0.001


def dscr(cash, ds):
    v = cash / ds
    q = f"Cash available for debt service is {money(cash)} and annual debt service is {money(ds)}. What is the debt service coverage ratio (DSCR)?"
    r = f"DSCR = {money(cash)} ÷ {money(ds)} = {v:.2f}x."
    return q, r, round(v, 2), "number", 0.005


def bond(face, coupon, ytm, years):
    p = bond_price(face, coupon, ytm, years)
    q = (f"A {years}-year bond with a {money(face)} face value pays a {coupon:.0%} annual coupon once a year. "
         f"If the yield to maturity is {ytm:.0%}, what is the bond's price?")
    r = (f"Annual coupon = {money(face * coupon)}. Price = present value of {years} coupons plus face value at {ytm:.0%} = {money(p)}.")
    return q, r, round(p, 2), "number", 0.001


def fifo_cogs(begin_units, begin_cost, buy_units, buy_cost, sold):
    from_begin = min(sold, begin_units)
    from_buy = sold - from_begin
    v = from_begin * begin_cost + from_buy * buy_cost
    q = (f"Beginning inventory is {begin_units} units at {money(begin_cost)} each. The company buys {buy_units} more units at "
         f"{money(buy_cost)} each, then sells {sold} units. Using FIFO (first in, first out), what is cost of goods sold?")
    r = (f"FIFO uses the oldest units first: {from_begin} × {money(begin_cost)} + {from_buy} × {money(buy_cost)} = {money(v)}.")
    return q, r, round(v, 2), "number", 0.001


def break_even(fixed, price, var):
    v = fixed / (price - var)
    q = (f"Fixed costs are {money(fixed)}. Each unit sells for {money(price)} and has variable cost of {money(var)}. "
         f"How many units must be sold to break even?")
    r = f"Contribution margin per unit = {money(price)} − {money(var)} = {money(price - var)}. Break-even units = {money(fixed)} ÷ {money(price - var)} = {v:,.0f}."
    return q, r, round(v, 0), "number", 0.001


def choice(question, options, correct, why):
    letters = "ABCD"
    opts = " ".join(f"({letters[i]}) {o}" for i, o in enumerate(options))
    q = f"{question} {opts}"
    r = why
    return q, r, correct, "choice", 0.0


TRAIN = [
    ("TR-001", "tax", corp_tax(2_400_000, 0.21, 0.06)),
    ("TR-002", "tax", effective_rate(1_050_000, 4_200_000)),
    ("TR-003", "accounting", straight_line_book_value(500_000, 50_000, 5, 3)),
    ("TR-004", "corporate_finance", after_tax_cost_of_debt(0.08, 0.21)),
    ("TR-005", "corporate_finance", cagr(12_000_000, 27_000_000, 4)),
    ("TR-006", "corporate_finance", loan_payment(1_000_000, 0.06, 5)),
    ("TR-007", "corporate_finance", project_npv(0.10, 1_000_000, 300_000, 5)),
    ("TR-008", "venture", post_money_safe(500_000, 10_000_000)),
    ("TR-009", "venture", convertible_note(1_000_000, 0.06, 18, 0.20, 8_000_000, 5_000_000, 2.50)),
    ("TR-010", "venture", dilution(4_000_000, 10_000_000, 2_500_000)),
    ("TR-011", "venture", option_pool_shuffle(20_000_000, 5_000_000, 0.10, 8_000_000)),
    ("TR-012", "accounting", current_ratio(3_400_000, 2_100_000)),
    ("TR-013", "accounting", ebitda(2_000_000, 500_000, 600_000, 900_000)),
    ("TR-014", "tax", cap_gains_tax(40_000, 100_000, 0.15, 0.038)),
    ("TR-015", "fund", tvpi(50_000_000, 70_000_000, 40_000_000)),
    ("TR-016", "fund", european_waterfall(100_000_000, 0.08, 3, 150_000_000, 0.20)),
    ("TR-017", "credit", dscr(5_000_000, 3_200_000)),
    ("TR-018", "corporate_finance", bond(1_000, 0.05, 0.06, 5)),
    ("TR-019", "accounting", fifo_cogs(100, 10, 200, 12, 250)),
    ("TR-020", "accounting", break_even(120_000, 50, 30)),
    ("TR-021", "accounting", choice(
        "In which section of the cash flow statement is a purchase of equipment reported?",
        ["Operating activities", "Investing activities", "Financing activities", "It is not reported"], "B",
        "Buying long-lived assets such as equipment is an investing activity.")),
    ("TR-022", "tax", choice(
        "Under US federal income tax rules, what holding period makes a gain on stock a long-term capital gain?",
        ["More than 6 months", "One year or less", "More than one year", "More than 5 years"], "C",
        "A capital asset must be held for more than one year for the gain to be long-term.")),
]

EVAL = [
    ("EV-001", "tax", corp_tax(3_750_000, 0.21, 0.05)),
    ("EV-002", "corporate_finance", cagr(8_000_000, 20_000_000, 5)),
    ("EV-003", "corporate_finance", loan_payment(2_500_000, 0.075, 10)),
    ("EV-004", "venture", post_money_safe(750_000, 12_000_000)),
    ("EV-005", "venture", convertible_note(500_000, 0.08, 24, 0.15, 6_000_000, 4_000_000, 1.80)),
    ("EV-006", "venture", option_pool_shuffle(30_000_000, 10_000_000, 0.15, 12_000_000)),
    ("EV-007", "fund", european_waterfall(200_000_000, 0.08, 4, 340_000_000, 0.20)),
    ("EV-008", "accounting", ebitda(3_200_000, 800_000, 1_100_000, 1_500_000)),
    ("EV-009", "corporate_finance", bond(1_000, 0.04, 0.05, 7)),
    ("EV-010", "accounting", choice(
        "Under FIFO (first in, first out), when purchase prices are rising, how does cost of goods sold compare with LIFO (last in, first out)?",
        ["FIFO cost of goods sold is lower", "FIFO cost of goods sold is higher", "They are always equal", "It cannot be determined"], "A",
        "With rising prices, FIFO expenses the older, cheaper units first, so cost of goods sold is lower than under LIFO.")),
]


def record(rid, category, built, split):
    q, r, a, t, tol = built
    if isinstance(a, float) and a.is_integer():
        a = int(a)
    return {
        "id": rid,
        "split": split,
        "category": category,
        "question": q,
        "context": "",
        "reasoning": r,
        "answer": a,
        "answer_type": t,
        "tolerance": tol,
        "difficulty": "easy" if t == "choice" else "medium",
        "source": "Synthetic; numbers invented and answers computed in tools/build_seed_data.py",
        "license": "Original work",
        "reviewed_by": "",
    }


def main():
    DATA.mkdir(exist_ok=True)
    train = [record(*row, "train") for row in TRAIN]
    evals = [record(*row, "eval") for row in EVAL]
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        with open(DATA / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = list(train[0].keys())
    with open(DATA / "seed_examples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in train + evals:
            w.writerow(row)
    print(f"Wrote {len(train)} train and {len(evals)} eval examples to {DATA}")


if __name__ == "__main__":
    main()
