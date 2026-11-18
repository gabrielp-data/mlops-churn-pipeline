"""Baixa a amostra pública IBM sem substituir o CSV versionado original."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = (
    "https://raw.githubusercontent.com/IBM/watsonx-ai-samples/"
    "e7345a897d3a440e0e8ee465150fe895bd5b29cb/"
    "cpd4.5/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv"
)
EXPECTED_MD5 = "3a27ec1ff5c0a90dd61cefa8a13184ab"
DESTINATION = Path("data/telco-public.csv")


def fetch_public_data(destination: Path = DESTINATION):
    if destination.exists():
        payload = destination.read_bytes()
        if hashlib.md5(payload).hexdigest() != EXPECTED_MD5:
            raise ValueError(
                f"Arquivo existente difere da amostra pública: {destination}"
            )
    else:
        with urlopen(URL, timeout=30) as response:
            payload = response.read()
        if hashlib.md5(payload).hexdigest() != EXPECTED_MD5:
            raise ValueError("Checksum da amostra pública não confere.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return destination


if __name__ == "__main__":
    print(fetch_public_data())
