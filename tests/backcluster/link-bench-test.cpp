#include "../../ios/Backburner/Sidecar/LinkBench.h"
#include <cassert>
#include <thread>
#include <unistd.h>

int main() {
    using namespace bc_link;
    size_t n = 0;
    assert(parse("linkbench 1 16384", n) && n == 16384);
    for (const char *s : {"linkbench 2 12", "linkbench 1 0", "linkbench 1 01", "linkbench 1 4194305", "linkbench 1 -1", "linkbench 1 12 trailing"})
        assert(!parse(s, n));
    for (size_t bytes : {size_t(16384), size_t(1048576), size_t(2662400), MAX_BYTES}) {
        int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        bool ok = false;
        std::thread worker([&] { ok = serve(fd[1], "linkbench 1 " + std::to_string(bytes)); close(fd[1]); });
        const auto deadline = clock::now() + std::chrono::seconds(5);
        auto line = [&]() {
            std::string s; char c;
            do { assert(transfer(fd[0], &c, 1, false, deadline)); s.push_back(c); } while (c != '\n');
            return s;
        };
        assert(line().find("\"bytes\":" + std::to_string(bytes)) != std::string::npos);
        std::vector<unsigned char> payload(bytes), reply(bytes);
        for (size_t i = 0; i < bytes; ++i) payload[i] = static_cast<unsigned char>(i % 251);
        assert(transfer(fd[0], payload.data(), bytes, true, deadline));
        assert(line() == "{\"received_bytes\":" + std::to_string(bytes) + "}\n");
        assert(transfer(fd[0], reply.data(), bytes, false, deadline));
        assert(reply == payload);
        close(fd[0]); worker.join(); assert(ok);
    }
    int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    unsigned char c;
    auto start = clock::now();
    assert(!transfer(fd[0], &c, 1, false, start + std::chrono::milliseconds(50)));
    assert(clock::now() - start < std::chrono::seconds(1));
    close(fd[0]); close(fd[1]);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    bool ok = true;
    std::thread worker([&] { ok = serve(fd[1], "linkbench 1 4"); close(fd[1]); });
    char r;
    do { assert(recv(fd[0], &r, 1, 0) == 1); } while (r != '\n');
    unsigned char bad[4] = {9, 9, 9, 9};
    assert(send(fd[0], bad, 4, 0) == 4);
    assert(recv(fd[0], &r, 1, 0) == 0);
    close(fd[0]); worker.join(); assert(!ok);
    std::puts("native link bench: framing, 4 payload sizes, integrity and deadline passed");
}
