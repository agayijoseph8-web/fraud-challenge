"""Interface Streamlit pour presenter les resultats au jury."""

from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_detection import detect_fraud, load_transactions

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_transactions.csv"
UNKNOWN = "Indisponible"


def render_interface(transactions: list[dict], results: list[dict]) -> None:
    """Affiche une analyse lisible des transactions et des alertes."""
    data = _build_dataframe(transactions, results)
    if data.empty:
        st.info("Aucune transaction a analyser.")
        return

    _render_decision_summary(data)
    filtered = _render_filters(data)

    if filtered.empty:
        st.warning("Aucune transaction ne correspond aux filtres actifs.")
        return

    overview_tab, alerts_tab, clients_tab, patterns_tab = st.tabs(
        ["Synthese", "File d'alertes", "Clients", "Signaux"]
    )

    with overview_tab:
        _render_overview(filtered)

    with alerts_tab:
        _render_alert_queue(data)

    with clients_tab:
        _render_clients(data)

    with patterns_tab:
        _render_patterns(data)


def _build_dataframe(transactions: list[dict], results: list[dict]) -> pd.DataFrame:
    tx_df = pd.DataFrame(transactions or [])
    result_df = pd.DataFrame(results or [])
    if tx_df.empty:
        return pd.DataFrame()

    for column in (
        "transaction_id",
        "timestamp",
        "user_id",
        "amount",
        "currency",
        "merchant",
        "country",
        "card_present",
    ):
        if column not in tx_df:
            tx_df[column] = None

    for column, default in (
        ("fraud_score", 0.0),
        ("is_suspicious", False),
        ("reason", "Transaction conforme au profil du client"),
    ):
        if column not in result_df:
            result_df[column] = default

    data = pd.concat(
        [
            tx_df.reset_index(drop=True),
            result_df[["fraud_score", "is_suspicious", "reason"]].reset_index(drop=True),
        ],
        axis=1,
    )
    data["fraud_score"] = pd.to_numeric(data["fraud_score"], errors="coerce").fillna(0.0)
    data["fraud_score"] = data["fraud_score"].clip(lower=0.0, upper=1.0)
    data["is_suspicious"] = data["is_suspicious"].fillna(False).astype(bool)
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["risk_level"] = data["fraud_score"].apply(_risk_level)
    data["decision"] = data["is_suspicious"].map(
        {True: "A verifier", False: "Conforme"}
    )
    data["user_display"] = data["user_id"].apply(_display_value)
    data["merchant_display"] = data["merchant"].apply(_display_value)
    data["country_display"] = data["country"].apply(_display_value)
    data["currency_display"] = data["currency"].apply(_display_value)
    data["card_present_display"] = data["card_present"].apply(_card_present_label)
    data["amount_display"] = data.apply(
        lambda row: _format_amount(row["amount"], row["currency_display"]),
        axis=1,
    )
    data["time_display"] = data["timestamp_dt"].apply(_format_timestamp)
    data["search_text"] = data.apply(_search_text, axis=1)
    return data


def _render_decision_summary(data: pd.DataFrame) -> None:
    total = len(data)
    suspicious = int(data["is_suspicious"].sum())
    critical = int((data["fraud_score"] >= 0.85).sum())
    avg_score = float(data["fraud_score"].mean())
    fraud_rate = suspicious / total if total else 0.0

    top_alert = data.sort_values("fraud_score", ascending=False).iloc[0]
    if suspicious:
        action = (
            f"Priorite: verifier {top_alert['transaction_id']} "
            f"avant les autres dossiers."
        )
    else:
        action = "Aucune alerte active dans ce lot."

    if suspicious:
        st.warning(action)
    else:
        st.success(action)
    st.caption(
        "Le moteur combine anomalies de montant, donnees manquantes, frequence, "
        "doublons et coherence geographique."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Transactions", total, "lot analyse")
    col_b.metric("Alertes", suspicious, f"{fraud_rate:.0%} du lot")
    col_c.metric("Critiques", critical, "score >= 0.85")
    col_d.metric("Score moyen", f"{avg_score:.2f}", "risque global")


def _render_filters(data: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filtres d'investigation", expanded=True):
        col_a, col_b, col_c, col_d = st.columns([1.5, 1.1, 1.1, 1.1])
        search = col_a.text_input("Recherche", placeholder="transaction, client, pays, marchand")
        min_score = col_b.slider("Score minimum", 0.0, 1.0, 0.0, 0.05)
        risk_options = ["Normal", "A surveiller", "A traiter", "Critique"]
        risks = col_c.multiselect("Niveau", risk_options, default=risk_options)
        mode = col_d.radio("Verdict", ["Tous", "Alertes", "Conformes"], horizontal=True)

        country_options = ["Tous"] + _sorted_values(data["country_display"])
        user_options = ["Tous"] + _sorted_values(data["user_display"])
        merchant_options = ["Tous"] + _sorted_values(data["merchant_display"])
        col_e, col_f, col_g = st.columns(3)
        country = col_e.selectbox("Pays", country_options)
        user = col_f.selectbox("Client", user_options)
        merchant = col_g.selectbox("Marchand", merchant_options)

    filtered = data[data["fraud_score"] >= min_score].copy()
    filtered = filtered[filtered["risk_level"].isin(risks)]
    if mode == "Alertes":
        filtered = filtered[filtered["is_suspicious"]]
    elif mode == "Conformes":
        filtered = filtered[~filtered["is_suspicious"]]
    if country != "Tous":
        filtered = filtered[filtered["country_display"] == country]
    if user != "Tous":
        filtered = filtered[filtered["user_display"] == user]
    if merchant != "Tous":
        filtered = filtered[filtered["merchant_display"] == merchant]
    if search.strip():
        query = search.strip().lower()
        filtered = filtered[filtered["search_text"].str.contains(query, na=False)]
    return filtered.sort_values(["fraud_score", "timestamp_dt"], ascending=[False, True])


def _render_overview(data: pd.DataFrame) -> None:
    left, right = st.columns([2.2, 1])
    with left:
        st.subheader("Transactions analysees")
        st.dataframe(
            _display_columns(data),
            width="stretch",
            hide_index=True,
            height=430,
            column_config=_transaction_table_config(),
        )

    with right:
        st.subheader("Repartition du risque")
        risk_counts = (
            data["risk_level"]
            .value_counts()
            .reindex(["Normal", "A surveiller", "A traiter", "Critique"], fill_value=0)
        )
        st.bar_chart(risk_counts)

        st.subheader("Raisons dominantes")
        reasons = _reason_counts(data)
        if reasons.empty:
            st.caption("Aucun signal suspect dans la selection.")
        else:
            st.dataframe(reasons, width="stretch", hide_index=True)


def _render_alert_queue(data: pd.DataFrame) -> None:
    alerts = data[data["is_suspicious"]].sort_values(
        ["fraud_score", "amount"], ascending=[False, False]
    )
    if alerts.empty:
        st.success("Aucune alerte active.")
        return

    st.subheader("Dossier prioritaire")
    options = alerts["transaction_id"].astype(str).tolist()
    selected_id = st.selectbox("Transaction", options)
    row = alerts[alerts["transaction_id"].astype(str) == selected_id].iloc[0]

    _render_alert_detail(row)
    st.progress(float(row["fraud_score"]))

    detail_left, detail_right = st.columns([1.2, 1])
    with detail_left:
        st.subheader("Contexte de transaction")
        detail = pd.DataFrame(
            [
                ("Montant", row["amount_display"]),
                ("Client", row["user_display"]),
                ("Marchand", row["merchant_display"]),
                ("Pays", row["country_display"]),
                ("Carte presente", row["card_present_display"]),
                ("Horodatage", row["time_display"]),
            ],
            columns=["Champ", "Valeur"],
        )
        st.dataframe(detail, width="stretch", hide_index=True)

    with detail_right:
        st.subheader("Transactions du meme client")
        peer_rows = data[data["user_display"] == row["user_display"]].sort_values(
            "timestamp_dt"
        )
        st.dataframe(
            _display_columns(peer_rows),
            width="stretch",
            hide_index=True,
            height=270,
            column_config=_transaction_table_config(compact=True),
        )

    st.subheader("Alertes a traiter")
    for index, (_, alert) in enumerate(alerts.head(6).iterrows(), start=1):
        _render_queue_item(index, alert)


def _render_clients(data: pd.DataFrame) -> None:
    st.subheader("Risque par client")
    clients = (
        data.groupby("user_display", dropna=False)
        .agg(
            transactions=("transaction_id", "count"),
            alertes=("is_suspicious", "sum"),
            score_max=("fraud_score", "max"),
            score_moyen=("fraud_score", "mean"),
            montant_total=("amount", "sum"),
            pays=("country_display", _join_unique),
        )
        .reset_index()
        .sort_values(["score_max", "alertes", "transactions"], ascending=[False, False, False])
    )
    st.dataframe(
        clients,
        width="stretch",
        hide_index=True,
        column_config={
            "user_display": "Client",
            "transactions": "Transactions",
            "alertes": "Alertes",
            "score_max": st.column_config.ProgressColumn(
                "Score max", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "score_moyen": st.column_config.NumberColumn("Score moyen", format="%.2f"),
            "montant_total": st.column_config.NumberColumn("Montant total", format="%.2f"),
            "pays": "Pays observes",
        },
    )

    risky_clients = clients[clients["alertes"] > 0]
    if not risky_clients.empty:
        st.subheader("Clients a revoir en premier")
        for _, row in risky_clients.head(4).iterrows():
            with st.container(border=True):
                st.write(f"**{row['user_display']}**")
                st.write(
                    f"{int(row['alertes'])} alerte(s), "
                    f"score max {float(row['score_max']):.2f}"
                )
                st.caption(str(row["pays"]))


def _render_patterns(data: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Concentration par pays")
        countries = (
            data.groupby("country_display", dropna=False)
            .agg(
                transactions=("transaction_id", "count"),
                alertes=("is_suspicious", "sum"),
                score_max=("fraud_score", "max"),
            )
            .reset_index()
            .sort_values(["score_max", "alertes"], ascending=[False, False])
        )
        st.dataframe(countries, width="stretch", hide_index=True)

    with right:
        st.subheader("Concentration par marchand")
        merchants = (
            data.groupby("merchant_display", dropna=False)
            .agg(
                transactions=("transaction_id", "count"),
                alertes=("is_suspicious", "sum"),
                score_max=("fraud_score", "max"),
            )
            .reset_index()
            .sort_values(["score_max", "alertes"], ascending=[False, False])
        )
        st.dataframe(merchants, width="stretch", hide_index=True)

    dated = data.dropna(subset=["timestamp_dt"]).copy()
    if not dated.empty:
        st.subheader("Evolution temporelle")
        dated["date"] = dated["timestamp_dt"].dt.date
        timeline = (
            dated.groupby("date")
            .agg(score_moyen=("fraud_score", "mean"), alertes=("is_suspicious", "sum"))
            .reset_index()
            .set_index("date")
        )
        st.line_chart(timeline)


def _display_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "transaction_id",
        "time_display",
        "user_display",
        "amount_display",
        "merchant_display",
        "country_display",
        "card_present_display",
        "fraud_score",
        "risk_level",
        "decision",
        "reason",
    ]
    return data[[column for column in columns if column in data]].rename(
        columns={
            "transaction_id": "Transaction",
            "time_display": "Date",
            "user_display": "Client",
            "amount_display": "Montant",
            "merchant_display": "Marchand",
            "country_display": "Pays",
            "card_present_display": "Carte",
            "fraud_score": "Score",
            "risk_level": "Risque",
            "decision": "Decision",
            "reason": "Justification",
        }
    )


def _transaction_table_config(compact: bool = False) -> dict:
    config = {
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=0.0, max_value=1.0, format="%.2f"
        ),
        "Justification": st.column_config.TextColumn(
            "Justification", width="large" if not compact else "medium"
        ),
    }
    return config


def _reason_counts(data: pd.DataFrame) -> pd.DataFrame:
    alerts = data[data["is_suspicious"]]
    if alerts.empty:
        return pd.DataFrame()
    reasons = alerts["reason"].fillna("").str.split("; ").explode()
    reasons = reasons[reasons.str.strip() != ""]
    return (
        reasons.value_counts()
        .reset_index()
        .rename(columns={"reason": "Signal", "count": "Occurrences"})
    )


def _render_alert_detail(row: pd.Series) -> None:
    with st.container(border=True):
        st.subheader(
            f"Transaction {row['transaction_id']} - {row['risk_level']} "
            f"(score {float(row['fraud_score']):.2f})"
        )
        st.write(
            f"{row['amount_display']} chez {row['merchant_display']} - "
            f"{row['country_display']}"
        )
        for reason in _split_reasons(row["reason"]):
            st.write(f"- {reason}")


def _render_queue_item(index: int, row: pd.Series) -> None:
    with st.container(border=True):
        st.write(f"**#{index} - {row['transaction_id']}**")
        st.write(f"{row['risk_level']} - score {float(row['fraud_score']):.2f}")
        st.caption(str(row["reason"]))


def _split_reasons(reason: str) -> list[str]:
    if not isinstance(reason, str) or not reason.strip():
        return ["Aucune justification fournie"]
    return [item.strip() for item in reason.split(";") if item.strip()]


def _risk_level(score: float) -> str:
    if score >= 0.85:
        return "Critique"
    if score >= 0.65:
        return "A traiter"
    if score >= 0.4:
        return "A surveiller"
    return "Normal"


def _format_amount(amount, currency) -> str:
    if pd.isna(amount):
        return UNKNOWN
    suffix = "" if currency == UNKNOWN else f" {currency}"
    return f"{float(amount):,.2f}{suffix}"


def _format_timestamp(value) -> str:
    if pd.isna(value):
        return UNKNOWN
    return value.strftime("%Y-%m-%d %H:%M")


def _display_value(value) -> str:
    if pd.isna(value) or value is None or value == "":
        return UNKNOWN
    return str(value)


def _card_present_label(value) -> str:
    if pd.isna(value) or value is None:
        return UNKNOWN
    return "Oui" if bool(value) else "Non"


def _sorted_values(values: pd.Series) -> list[str]:
    return sorted(str(value) for value in values.dropna().unique())


def _join_unique(values: pd.Series) -> str:
    unique = sorted(str(value) for value in values.dropna().unique() if value != UNKNOWN)
    return ", ".join(unique) if unique else UNKNOWN


def _search_text(row: pd.Series) -> str:
    parts = [
        row.get("transaction_id"),
        row.get("user_display"),
        row.get("merchant_display"),
        row.get("country_display"),
        row.get("currency_display"),
        row.get("reason"),
    ]
    return " ".join(str(part).lower() for part in parts if not pd.isna(part))


def _transactions_key(transactions: list[dict]) -> tuple:
    return tuple(
        (
            tx.get("transaction_id"),
            tx.get("timestamp"),
            tx.get("user_id"),
            tx.get("amount"),
            tx.get("country"),
        )
        for tx in transactions
        if isinstance(tx, dict)
    )


def main() -> None:
    st.set_page_config(
        page_title="Detection de fraude - Hackathon INTELO2026",
        layout="wide",
    )

    st.title("Detection de fraude financiere")
    st.caption("Hackathon INTELO2026 - analyse automatique des transactions")

    with st.sidebar:
        st.header("Donnees")
        use_sample = st.toggle("Utiliser le fichier d'exemple", value=True)
        transactions: list[dict] = []

        if use_sample:
            transactions = load_transactions(str(SAMPLE_CSV))
            st.success(f"{len(transactions)} transactions chargees")
        else:
            uploaded = st.file_uploader("Importer un CSV", type=["csv"])
            if uploaded:
                tmp = Path(".streamlit_upload.csv")
                tmp.write_bytes(uploaded.getvalue())
                transactions = load_transactions(str(tmp))
                tmp.unlink(missing_ok=True)
                st.success(f"{len(transactions)} transactions importees")

        st.divider()
        st.markdown(
            "**Moteur :** score de risque, verdict et justification par transaction."
        )

    if not transactions:
        st.info("Chargez des transactions depuis la barre laterale.")
        return

    current_key = _transactions_key(transactions)
    if st.session_state.get("transactions_key") != current_key:
        st.session_state["transactions_key"] = current_key
        st.session_state.pop("analysis_results", None)
        st.session_state["auto_run_pending"] = True

    action_col, _ = st.columns([1, 3])
    with action_col:
        run_requested = st.button("Actualiser l'analyse", type="primary")

    if run_requested or st.session_state.pop("auto_run_pending", False):
        try:
            results = detect_fraud(transactions)
        except NotImplementedError:
            st.error("La fonction detect_fraud n'est pas encore implementee.")
            return
        except Exception as exc:
            st.error(f"Erreur pendant l'analyse : {exc}")
            return

        st.session_state["analysis_results"] = results

    results = st.session_state.get("analysis_results")
    if results is None:
        st.info("Cliquez sur Actualiser l'analyse pour afficher les resultats.")
        return

    render_interface(transactions, results)


if __name__ == "__main__":
    main()
