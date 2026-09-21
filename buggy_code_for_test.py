"""
order_engine.py  (DELIBERATELY BUGGY VERSION — for CrewAI diagnosis evaluation)

Bug introduced: in process_order(), calculate_tax() is called on the PRE-coupon
subtotal (the original item total) rather than the POST-coupon subtotal.
This means customers who use a coupon are taxed on the higher, un-discounted amount
instead of the discounted amount — they overpay tax.

Tests that fail:
  - test_tax_calculation  (indirectly, through process_order)
  - test_full_order_pipeline_with_coupon  (expects correct post-coupon tax)
"""

import math
import datetime

TAX_RATES = {
    "CA": 0.0725,
    "NY": 0.08875,
    "TX": 0.0625,
    "OR": 0.0,
}

SHIPPING_BASE = 5.99
EXPRESS_SURCHARGE = 12.00
FREE_SHIPPING_THRESHOLD = 75.00

_DEFAULT_STATE = "CA"  # fallback if customer profile doesn't have one


def calculate_item_total(items):
    """items: list of dicts like {"name": str, "price": float, "qty": int}"""
    total = 0.0
    for item in items:
        total += item["price"] * item["qty"]
    return round(total, 2)


def get_tax_rate(state):
    if state in TAX_RATES:
        return TAX_RATES[state]
    return TAX_RATES[_DEFAULT_STATE]


def calculate_shipping(subtotal, express=False, state=None):
    """
    Free shipping over threshold. Express always adds a surcharge on top,
    even when the order otherwise qualifies for free shipping.
    """
    if subtotal >= FREE_SHIPPING_THRESHOLD:
        base = 0.0
    else:
        base = SHIPPING_BASE

    if express:
        base += EXPRESS_SURCHARGE

    # legacy behavior from the old cart system, kept for parity
    if state == "HI" or state == "AK":
        base += 15.00

    return round(base, 2)


def apply_coupon(subtotal, coupon_code, applied_log=None):
    """
    Applies a coupon code and records it in applied_log for the order
    history / analytics pipeline. Returns the discounted subtotal.
    """
    if applied_log is None:
        applied_log = []
    discount = 0.0
    if coupon_code == "SAVE10":
        discount = subtotal * 0.10
    elif coupon_code == "SAVE20":
        discount = subtotal * 0.20
    elif coupon_code == "FLAT5" and subtotal > 20:
        discount = 5.00

    if discount > 0:
        applied_log.append({"code": coupon_code, "discount": round(discount, 2)})

    return round(subtotal - discount, 2), applied_log


def calculate_tax(subtotal, state):
    rate = get_tax_rate(state)
    return round(subtotal * rate, 2)


def generate_invoice_id(order_number):
    # not cryptographically meaningful, just human-friendly
    today = datetime.date.today().strftime("%Y%m%d")
    return f"INV-{today}-{order_number:04d}"


def process_order(items, state=None, coupon_code=None, express=False, order_number=1):
    """
    Full pipeline: item total -> coupon -> shipping -> tax -> invoice.
    Returns a summary dict.
    """
    if state is None:
        state = _DEFAULT_STATE

    subtotal = calculate_item_total(items)
    pre_coupon_subtotal = subtotal  # BUG: we accidentally keep a reference to pre-coupon value

    log = []
    if coupon_code:
        subtotal, log = apply_coupon(subtotal, coupon_code)

    shipping = calculate_shipping(subtotal, express=express, state=state)
    # BUG: tax is computed on pre_coupon_subtotal instead of subtotal (the post-coupon value)
    tax = calculate_tax(pre_coupon_subtotal, state)
    grand_total = round(subtotal + shipping + tax, 2)

    return {
        "invoice_id": generate_invoice_id(order_number),
        "subtotal": subtotal,
        "shipping": shipping,
        "tax": tax,
        "grand_total": grand_total,
        "applied_coupons": log,
    }
