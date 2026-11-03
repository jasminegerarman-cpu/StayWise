# Original historical data

This project downloads the public **Hotel Booking Demand** dataset the first time you run `START_STAYWISE.bat` or `python setup_project.py`.

- Expected original dataset: `hotel_bookings.csv` (119,390 historical booking records, 32 attributes).
- Public CSV: https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv
- Source authors: Nuno António, Ana de Almeida, Luis Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22:41-49. https://doi.org/10.1016/j.dib.2018.11.126

The CSV is **not included in the distributed ZIP**. First launch requires Internet; to set it up offline, separately put the *original* hotels.csv from the source above at `data/hotel_bookings.csv`. This project does **not** substitute simulated data to satisfy the minimum requirement.

The training script logs original and retained record counts and stores them in `models/evaluation.json`. Retained original booking observations are used for training; none are artificially duplicated.
