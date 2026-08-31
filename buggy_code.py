"""
order_engine.py

Order processing module for the checkout system. Handles cart totals,
coupon application, shipping tiers, tax, and invoice generation.

TODO: refactor calculate_shipping, it's grown a lot since the express-tier launch.
TODO: pull TAX_RATES out into config once finance signs off on the new schema.
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

    log = []
    if coupon_code:
        subtotal, log = apply_coupon(subtotal, coupon_code)

    shipping = calculate_shipping(subtotal, express=express, state=state)
    tax = calculate_tax(subtotal, state)
    grand_total = round(subtotal + shipping + tax, 2)

    return {
        "invoice_id": generate_invoice_id(order_number),
        "subtotal": subtotal,
        "shipping": shipping,
        "tax": tax,
        "grand_total": grand_total,
        "applied_coupons": log,
    }
