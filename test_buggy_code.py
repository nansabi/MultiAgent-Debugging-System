from buggy_code import (
    calculate_item_total,
    get_tax_rate,
    calculate_shipping,
    apply_coupon,
    calculate_tax,
    process_order,
)


def test_item_total_basic():
    items = [{"name": "mug", "price": 12.00, "qty": 2}]
    assert calculate_item_total(items) == 24.00


def test_item_total_multiple_items():
    items = [
        {"name": "mug", "price": 12.00, "qty": 2},
        {"name": "plate", "price": 8.50, "qty": 3},
    ]
    assert calculate_item_total(items) == 49.50


def test_tax_rate_known_state():
    assert get_tax_rate("TX") == 0.0625


def test_tax_rate_unknown_state_falls_back():
    assert get_tax_rate("ZZ") == 0.0725


def test_shipping_below_threshold():
    assert calculate_shipping(30.00) == 5.99


def test_shipping_free_over_threshold():
    assert calculate_shipping(100.00) == 0.0


def test_shipping_express_always_adds_surcharge():
    assert calculate_shipping(100.00, express=True) == 12.00


def test_shipping_hawaii_surcharge():
    assert calculate_shipping(30.00, state="HI") == 20.99


def test_tax_calculation():
    assert calculate_tax(100.00, "NY") == 8.88


def test_coupon_save10():
    discounted, log = apply_coupon(50.00, "SAVE10")
    assert discounted == 45.00
    assert log == [{"code": "SAVE10", "discount": 5.00}]


def test_coupon_no_code_leaves_log_empty():
    # A completely separate customer, later in the day, applies no coupon.
    # Their order history should show no coupons — this is a fresh order.
    discounted, log = apply_coupon(60.00, None)
    assert log == []


def test_full_order_pipeline_no_coupon():
    items = [{"name": "mug", "price": 12.00, "qty": 1}]
    result = process_order(items, state="TX", order_number=7)
    assert result["subtotal"] == 12.00
    assert result["applied_coupons"] == []


def test_full_order_pipeline_with_coupon():
    items = [{"name": "mug", "price": 50.00, "qty": 1}]
    result = process_order(items, state="CA", coupon_code="SAVE20", order_number=8)
    assert result["subtotal"] == 40.00
    assert result["applied_coupons"] == [{"code": "SAVE20", "discount": 10.00}]


def test_grand_total_is_sum_not_difference():
    # Regression test: mutation testing found that subtotal + shipping + tax
    # in process_order's grand_total was never independently verified. A
    # mutant flipping + to - passed the full suite undetected.
    items = [{"name": "mug", "price": 10.00, "qty": 1}]
    result = process_order(items, state="TX", express=False, order_number=1)
    expected = round(result["subtotal"] + result["shipping"] + result["tax"], 2)
    assert result["grand_total"] == expected
    assert result["grand_total"] > result["subtotal"]  # rules out a subtraction bug