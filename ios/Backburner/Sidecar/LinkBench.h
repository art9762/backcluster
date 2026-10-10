#pragma once
// Additive control-port benchmark. No model state, files or new listeners.
// Every transfer has one absolute deadline and a 4 MiB allocation ceiling.
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <string>
#include <vector>

namespace bc_link {
constexpr size_t MAX_BYTES = 4 * 1024 * 1024;
using clock = std::chrono::steady_clock;
inline bool transfer(int fd, void *buffer, size_t bytes, bool sending, clock::time_point deadline) {
    auto *p = static_cast<unsigned char *>(buffer);
    while (bytes) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - clock::now()).count();
        if (ms <= 0) return false;
        pollfd waiter{fd, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
        int ready = poll(&waiter, 1, static_cast<int>(ms));
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) return false;
        int flags = MSG_DONTWAIT;
#ifdef MSG_NOSIGNAL
        if (sending) flags |= MSG_NOSIGNAL;
#endif
        ssize_t n = sending ? send(fd, p, bytes, flags) : recv(fd, p, bytes, flags);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n <= 0) return false;
        p += n;
        bytes -= static_cast<size_t>(n);
    }
    return true;
}
inline bool write(int fd, const std::string &s, clock::time_point deadline) {
    return transfer(fd, const_cast<char *>(s.data()), s.size(), true, deadline);
}
inline bool parse(const std::string &line, size_t &bytes) {
    const std::string prefix = "linkbench 1 ";
    if (line.compare(0, prefix.size(), prefix) != 0) return false;
    const std::string number = line.substr(prefix.size());
    if (number.empty() || number.size() > 7 || number[0] == '0') return false;
    bytes = 0;
    for (char c : number) {
        if (c < '0' || c > '9') return false;
        bytes = bytes * 10 + static_cast<size_t>(c - '0');
    }
    return bytes > 0 && bytes <= MAX_BYTES;
}
inline bool serve(int fd, const std::string &line) {
    size_t bytes = 0;
    if (!parse(line, bytes)) return false;
#ifdef SO_NOSIGPIPE
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    const auto deadline = clock::now() + std::chrono::seconds(10);
    std::vector<unsigned char> buffer(bytes);
    const std::string ready = "{\"document_kind\":\"backcluster_link_ready\",\"version\":1,\"bytes\":" + std::to_string(bytes) + "}\n";
    if (!write(fd, ready, deadline) || !transfer(fd, buffer.data(), bytes, false, deadline)) return false;
    // Validate the canonical pattern before acknowledging; detect copy/framing corruption.
    for (size_t i = 0; i < bytes; ++i) if (buffer[i] != static_cast<unsigned char>(i % 251)) return false;
    const std::string ack = "{\"received_bytes\":" + std::to_string(bytes) + "}\n";
    return write(fd, ack, deadline) && transfer(fd, buffer.data(), bytes, true, deadline);
}
} // namespace bc_link
