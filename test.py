from cryptography.fernet import Fernet


def encrypt(raw_key: str) -> str:
    # ponytail: plug real encryption here (e.g. Fernet.encrypt) when a KMS
    # key/secret is available. Keeping this isolated per design rule #3.
    master_key = "J8ZvAsmsSCiEg_U25hylZQo7IVTyyAzGfXZY-DjQQ0U="
    fernet = Fernet(master_key)

    encrypted_key = fernet.encrypt(raw_key.encode())
    return encrypted_key.decode()


def decrypt(stored_value: str) -> str:
    # ponytail: plug real decryption here to match encrypt() above.

    master_key = "J8ZvAsmsSCiEg_U25hylZQo7IVTyyAzGfXZY-DjQQ0U="
    fernet = Fernet(master_key)

    return fernet.decrypt(stored_value).decode()


encrypt("sk-KEzLkDC9IkYiDlRRr4KjX0tvoaUsQDNw2gg0b88PgUJTVemSFGGNSOpc9ABZWNqO")
decrypt(
    "gAAAAABqhs0W2NIVGpRpOWKdPO-dEI-653zqo5NTRYAaNl4KrLAvgDIX7MypMcZfWJEyhPSdWiruHqx2WMlnAnkLIo7yZskSL3RIdDMthYs2iH5bB95FpbtWaA70tE_PJtdp_kE4ylux1bAllcWrSWZ6eeeRHitEcT69GwK1rw9mgNMk_wrQ_Nw="
)
