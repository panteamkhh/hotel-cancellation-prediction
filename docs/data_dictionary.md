# Data dictionary

Reference for the 32 columns in the raw **Hotel Booking Demand** dataset and
the 10 features this project derives from them.

Source: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle).
119,390 raw rows for a City Hotel and a Resort Hotel, 2015–2017.

## Raw columns

| Column | Type | Description |
|---|---|---|
| `hotel` | category | `City Hotel` or `Resort Hotel`. |
| `is_canceled` | 0/1 | **Target.** 1 if the booking was canceled. |
| `lead_time` | int | Days between booking date and arrival date. |
| `arrival_date_year` | int | Arrival year (2015–2017). |
| `arrival_date_month` | category | Arrival month name (`January`…`December`). |
| `arrival_date_week_number` | int | ISO week of arrival (1–53). |
| `arrival_date_day_of_month` | int | Day of month of arrival (1–31). |
| `stays_in_weekend_nights` | int | Weekend nights booked (Sat/Sun). |
| `stays_in_week_nights` | int | Week nights booked (Mon–Fri). |
| `adults` | int | Number of adults. |
| `children` | float | Number of children (4 missing values). |
| `babies` | int | Number of babies. |
| `meal` | category | Meal plan: `BB`, `HB`, `FB`, `SC`, `Undefined`. |
| `country` | category | Country of origin (ISO 3166-1 alpha-3). |
| `market_segment` | category | Booking market segment (e.g. `Online TA`, `Direct`). |
| `distribution_channel` | category | Distribution channel (e.g. `TA/TO`, `Direct`, `Corporate`). |
| `is_repeated_guest` | 0/1 | 1 if the guest has booked before. |
| `previous_cancellations` | int | Prior bookings canceled by this guest. |
| `previous_bookings_not_canceled` | int | Prior bookings completed by this guest. |
| `reserved_room_type` | category | Room type reserved (code letter). |
| `assigned_room_type` | category | Room type actually assigned. |
| `booking_changes` | int | Number of changes made to the booking. |
| `deposit_type` | category | `No Deposit`, `Non Refund`, `Refundable`. |
| `agent` | category | Travel agency ID, or missing for no agent. |
| `company` | category | Company ID; ~94% missing. |
| `days_in_waiting_list` | int | Days the booking sat on the waiting list. |
| `customer_type` | category | `Transient`, `Transient-Party`, `Contract`, `Group`. |
| `adr` | float | Average Daily Rate (revenue per occupied room-night). |
| `required_car_parking_spaces` | int | Parking spaces requested. |
| `total_of_special_requests` | int | Number of special requests. |
| `reservation_status` | category | `Check-Out`, `Canceled`, `No-Show`. **Leakage — dropped.** |
| `reservation_status_date` | date | Date the status was set. **Leakage — dropped.** |

## Derived features

Created in `src/feature_engineering.py` (notebook `08`).

| Feature | Definition | Notes |
|---|---|---|
| `total_members` | `adults + children + babies` | Rows equal to 0 are invalid bookings and are dropped. |
| `total_guests` | `babies + children + adults` | Same value as `total_members`; kept for readability. |
| `has_family` | `1` if `children > 0` or `babies > 0` | Family-booking indicator. |
| `total_stay` | `stays_in_weekend_nights + stays_in_week_nights` | Total nights. |
| `room_changed` | `reserved_room_type != assigned_room_type` | Proxy for an upgrade/reassignment. |
| `long_lead_time` | `lead_time > 90` | Threshold matches the risk analysis in notebook 03. |
| `arrival_quarter` | `arrival_date_week_number` binned into `Q1`–`Q4` | Bins are `(0, 13, 26, 39, 53]` so ISO week 53 is not lost. |
| `total_previous_bookings` | `previous_cancellations + previous_bookings_not_canceled` | Guest history size. |
| `prev_cancel_ratio` | `previous_cancellations / total_previous_bookings` | 0 when the guest has no history (guarded against divide-by-zero). |
| `revenue` | `adr * total_stay` | Derived proxy — the dataset has no revenue column. |

## Cleaning rules

Applied in `src/data_cleaning.py` (notebook `01`):

1. **Duplicates** — exact duplicate rows dropped (31,994 of them).
2. **Missing values** — `children` filled with the mode; `company` dropped
   (94% missing); `agent` filled with `"no agent"`; `country` filled with the mode.
3. **Invalid bookings** — rows with zero total guests removed.
4. **ADR outliers** — negative ADR removed, plus the top 0.1% of ADR values.
5. **Leakage** — `reservation_status` / `reservation_status_date` dropped before modeling.

Result: **87,139 clean rows** and a **27.5% cancellation rate** (vs. the
~37% figure you get on the raw, duplicate-inflated data).
