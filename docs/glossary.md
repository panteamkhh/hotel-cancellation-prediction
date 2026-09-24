# Glossary

| Term | Meaning |
|---|---|
| **ADR** (Average Daily Rate) | Average revenue per occupied room-night, `adr` in the dataset. |
| **Cancellation rate** | Share of bookings where `is_canceled = 1`. |
| **Lead time** | Days between the booking date and the arrival date. The project's strongest single predictor. |
| **ADR outlier** | Negative ADR values and the top 0.1% of ADR, removed during cleaning. |
| **Target leakage** | Using information only known after the outcome (e.g. `reservation_status`) as a feature; excluded from modeling. |
| **Distribution channel** | How the booking reached the hotel: `TA/TO`, `Direct`, `Corporate`, `GDS`, `Undefined`. |
| **Market segment** | Finer-grained booking source group (e.g. `Online TA`, `Offline TA/TO`, `Groups`, `Direct`). |
| **TA/TO** | Travel agent / tour operator — the dominant channel by both volume and revenue. |
| **Customer type** | `Transient`, `Transient-Party`, `Contract` or `Group`. |
| **Deposit type** | `No Deposit`, `Non Refund` or `Refundable`. `Non Refund` cancels most often here. |
| **ADR proxy revenue** | `revenue = adr * total_stay`; the dataset has no true revenue field. |
| **ROC-AUC** | Area under the ROC curve — the probability a random positive ranks above a random negative. |
| **Precision / Recall / F1** | Classification metrics reported for every model in the comparison table. |
| **Cost-sensitive threshold** | A decision cutoff chosen to minimise expected business cost rather than using 0.5. |
| **Stratified split** | Train/test split that preserves the class ratio; used so the 27.5% positive rate holds in both sets. |
| **One-hot encoding** | Turning categorical columns into 0/1 indicator columns (573 features in total). |
| **ISO week 53** | The extra week some years have; `arrival_quarter` bins to `(0, 53]` so it is not lost. |
