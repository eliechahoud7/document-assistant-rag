"""Identité de l'utilisateur de l'interface web : seul point de lecture de ``st.user``.

``st.user`` est construit par le serveur Streamlit à partir d'un cookie signé avec
``[auth].cookie_secret`` (connexion Google via OpenID Connect) : le navigateur ne peut ni le
choisir ni le modifier, et le code de l'application ne peut pas l'écrire. Rien ici ne lit
``st.session_state`` ni les paramètres de l'adresse : l'identité est relue à chaque exécution
de la page. Les tests remplacent ``signed_in_claims`` (aucune connexion réelle hors ligne).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping

import streamlit as st

from doc_assistant.services.user_libraries import UserIdentity


def signed_in_claims() -> Mapping[str, object] | None:
    """Informations de connexion vérifiées ; ``None`` si personne n'est connecté."""
    if not st.user.is_logged_in:
        return None
    return dict(st.user.to_dict())


def current_user() -> UserIdentity | None:
    """Utilisateur connecté ; lève ``AuthenticationError`` si l'identité est incomplète."""
    claims = signed_in_claims()
    return None if claims is None else UserIdentity.from_claims(claims)


def login_dependency_available() -> bool:
    """``st.login`` exige Authlib (``streamlit[auth]``)."""
    return importlib.util.find_spec("authlib") is not None
