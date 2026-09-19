# Raw data

`hotel_bookings.csv` — the raw **Hotel Booking Demand** dataset
(119,390 rows, 32 columns) from Kaggle:

https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

The file is committed to this repository (16 MB) so the whole project is
reproducible without a manual download. If you want to refresh it, download
the latest CSV from Kaggle and replace `data/raw/hotel_bookings.csv`.

`src/data_loader.py` validates that the file contains all 32 expected columns
before anything else runs.
