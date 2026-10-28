"""Regression checks for Windows CSV limits and isolated Streamlit transports."""

from __future__ import annotations

import asyncio
import socket

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("host", ["192.0.2.1", "8.8.8.8", "0.0.0.0", "::",
                                 "2001:db8::1", "::ffff:192.0.2.1", "example.com",
                                 "localhost", "127.0.0.1.example.com"])
@pytest.mark.parametrize("method", ["connect", "connect_ex", "create_connection", "getaddrinfo", "sendto"])
def test_streamlit_network_guard_blocks_external_wildcard_and_dns(host, method, streamlit_loopback):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        with pytest.raises(AssertionError, match="blocked"):
            if method == "create_connection":
                socket.create_connection((host, 443), timeout=.01)
            elif method == "getaddrinfo":
                socket.getaddrinfo(host, 443)
            elif method == "sendto":
                stream.sendto(b"synthetic", (host, 443))
            else:
                getattr(stream, method)((host, 443))


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_only_streamlit_tests_can_resolve_numeric_loopback(host, streamlit_loopback):
    resolved = socket.getaddrinfo(host, 80)
    assert resolved
    assert all(item[4][0] == host for item in resolved)


@pytest.mark.parametrize("method", ["connect", "connect_ex", "create_connection"])
def test_streamlit_loopback_tcp_really_works(method, streamlit_loopback):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2)
        address = listener.getsockname()
        if method == "create_connection":
            client = socket.create_connection(address, timeout=2)
        else:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(2)
            result = getattr(client, method)(address)
            if method == "connect_ex":
                assert result == 0
        with client, listener.accept()[0] as peer:
            client.sendall(b"synthetic")
            assert peer.recv(32) == b"synthetic"


def test_streamlit_asyncio_self_pipe_and_apptest_start(streamlit_loopback):
    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(asyncio.sleep(0, result="offline")) == "offline"
    finally:
        loop.close()
    app = AppTest.from_string("import streamlit as st\nst.write('synthetic offline UI')").run()
    assert not app.exception
    assert app.markdown[0].value == "synthetic offline UI"


@pytest.mark.parametrize("method", ["connect", "connect_ex", "create_connection", "getaddrinfo"])
def test_non_ui_tests_still_block_even_loopback(method, no_external_network):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        with pytest.raises(AssertionError, match="blocked"):
            if method == "create_connection":
                socket.create_connection(("127.0.0.1", 80), timeout=.01)
            elif method == "getaddrinfo":
                socket.getaddrinfo("127.0.0.1", 80)
            else:
                getattr(stream, method)(("127.0.0.1", 80))


@pytest.mark.parametrize("method", ["gethostbyname", "gethostbyname_ex", "gethostbyaddr"])
def test_dns_never_escapes_the_loopback_exception(method, streamlit_loopback):
    with pytest.raises(AssertionError, match="DNS"):
        getattr(socket, method)("example.com")


@pytest.mark.parametrize("value", ["cach_ly", ["cach_ly"]])
def test_query_parameter_scalar_and_singleton_list_are_equivalent(value, query_param_values):
    assert query_param_values(value) == ["cach_ly"]


@pytest.mark.parametrize("value", [[], ["cach_ly", "thu_rac"], ["cach_ly", "cach_ly"]])
def test_query_parameter_normalization_preserves_wrong_or_multiple_values(value, query_param_values):
    assert query_param_values(value) != ["cach_ly"]


@pytest.mark.parametrize("value", [None, 7, ("cach_ly",), [7]])
def test_invalid_query_parameter_shapes_are_not_coerced(value, query_param_values):
    with pytest.raises(AssertionError):
        query_param_values(value)
