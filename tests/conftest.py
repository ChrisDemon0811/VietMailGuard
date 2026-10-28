"""Shared test compatibility and explicit network isolation."""

from __future__ import annotations

import ipaddress
import socket

import pytest


def _is_loopback(address: object) -> bool:
    if not isinstance(address, tuple) or len(address) < 2:
        return False
    try:
        host = ipaddress.ip_address(address[0])
    except (TypeError, ValueError):
        return False
    mapped = getattr(host, "ipv4_mapped", None)
    return host.is_loopback or bool(mapped and mapped.is_loopback)


@pytest.fixture
def no_external_network(request, monkeypatch):
    """Block network I/O; only explicit Streamlit tests may use numeric loopback."""
    allow_loopback = "streamlit_loopback" in request.fixturenames
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_sendto = socket.socket.sendto

    def check(address):
        if not (allow_loopback and _is_loopback(address)):
            raise AssertionError("External network and non-numeric hostnames are blocked in tests")

    def connect(connection, address):
        check(address)
        if connection.family not in {socket.AF_INET, socket.AF_INET6}:
            raise AssertionError("Only IP loopback sockets are allowed for Streamlit")
        return original_connect(connection, address)

    def connect_ex(connection, address):
        check(address)
        if connection.family not in {socket.AF_INET, socket.AF_INET6}:
            raise AssertionError("Only IP loopback sockets are allowed for Streamlit")
        return original_connect_ex(connection, address)

    def create_connection(address, *args, **kwargs):
        check(address)
        return original_create_connection(address, *args, **kwargs)

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        check((host, port))
        return original_getaddrinfo(host, port, family, type, proto,
                                    flags | socket.AI_NUMERICHOST)

    def sendto(connection, data, *args):
        check(args[-1] if args else None)
        return original_sendto(connection, data, *args)

    def no_dns(*args, **kwargs):
        raise AssertionError("DNS lookups are blocked in isolated tests")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", no_dns)
    monkeypatch.setattr(socket, "gethostbyname_ex", no_dns)
    monkeypatch.setattr(socket, "gethostbyaddr", no_dns)


@pytest.fixture
def streamlit_loopback(no_external_network):
    """Opt in to the loopback self-pipe required by Streamlit/asyncio on Windows."""


@pytest.fixture
def query_param_values():
    """Normalize scalar/list AppTest representations without losing multiplicity."""
    def values(value):
        if isinstance(value, str):
            return [value]
        assert isinstance(value, list) and all(isinstance(item, str) for item in value)
        return value

    return values
