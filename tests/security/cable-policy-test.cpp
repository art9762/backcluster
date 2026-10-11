// cable-policy-test.cpp - the phone app's connection gate and argument checks (ios/Backburner/Sidecar/CableOnly.h), on the Mac.
//   tests/security/run.sh   (builds and runs this; exits non-zero on any failure)
// Covers: who may connect (cable / loopback allowed; Wi-Fi, cellular, routable, mismatched families refused), the escape-hatch
// list, live sockets through cable_accept(), the servers' default loopback-only filters, and the fetch / path validators.
#include "../../ios/Backburner/Sidecar/CableOnly.h"
#include "../../phone-attn/phone-attn.h"
#include "tools/split-prefill/tail-server.h"

#include <cstdio>
#include <thread>

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, msg) do { if (cond) { g_pass++; } else { g_fail++; fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, msg); } } while (0)

static sockaddr_in v4(const char * ip, int port = 0) {
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_len = sizeof a; a.sin_port = htons(port); inet_pton(AF_INET, ip, &a.sin_addr); return a;
}
static sockaddr_in6 v6(const char * ip, uint32_t scope = 0) {
    sockaddr_in6 a = {}; a.sin6_family = AF_INET6; a.sin6_len = sizeof a; inet_pton(AF_INET6, ip, &a.sin6_addr); a.sin6_scope_id = scope; return a;
}
static bool allow(const sockaddr * l, const sockaddr * p, uint32_t t, const char * env = nullptr) {
    std::string why; return bb::decide(l, p, t, env, why) == bb::verdict::allow;
}
#define SA(x) ((const sockaddr *) &(x))

static void test_decide() {
    const uint32_t WIRED = IFRTYPE_FUNCTIONAL_WIRED, WIFI = IFRTYPE_FUNCTIONAL_WIFI_INFRA, CELL = IFRTYPE_FUNCTIONAL_CELLULAR,
                   AWDL = IFRTYPE_FUNCTIONAL_WIFI_AWDL, UNK = ~0u;
    auto ll1 = v4("169.254.10.2"), ll2 = v4("169.254.77.9"), lo1 = v4("127.0.0.1"), lo2 = v4("127.0.0.1"),  // sensitive-ok: test fixtures
         lan1 = v4("192.168.1.20"), lan2 = v4("192.168.1.5"), pub = v4("8.8.8.8");  // sensitive-ok: test fixtures
    CHECK(allow(SA(ll1), SA(ll2), WIRED), "cable link-local over a wired interface is allowed");
    CHECK(allow(SA(lo1), SA(lo2), UNK), "loopback is allowed (in-app hops: the RPC gate, the Wi-Fi tunnel)");
    CHECK(!allow(SA(ll1), SA(ll2), WIFI), "self-assigned 169.254 on Wi-Fi is refused");
    CHECK(!allow(SA(ll1), SA(ll2), AWDL), "AWDL is refused");
    CHECK(!allow(SA(ll1), SA(ll2), CELL), "cellular is refused");
    CHECK(!allow(SA(ll1), SA(ll2), UNK), "an interface of unknown type is refused (fail closed)");
    CHECK(!allow(SA(lan1), SA(lan2), WIRED), "a routable LAN address is refused even on a wired interface");
    CHECK(!allow(SA(lan1), SA(lan2), WIFI), "Wi-Fi LAN is refused");
    CHECK(!allow(SA(ll1), SA(pub), WIRED), "a routable peer on the cable interface is refused");
    CHECK(!allow(SA(lo1), SA(ll2), WIRED), "loopback local with a non-loopback peer is refused");
    CHECK(!allow(SA(ll1), SA(lo1), WIRED), "link-local local with a loopback peer is refused");

    auto l6 = v6("fe80::1", 7), p6 = v6("fe80::abcd", 7), lo6 = v6("::1"), g6 = v6("2001:db8::1"), m4l = v6("::ffff:169.254.1.1"),  // sensitive-ok: test fixtures
         m4p = v6("::ffff:169.254.2.2"), m4lan = v6("::ffff:192.168.1.9");  // sensitive-ok: test fixtures
    CHECK(allow(SA(l6), SA(p6), WIRED), "IPv6 link-local on the cable is allowed");
    CHECK(!allow(SA(l6), SA(p6), WIFI), "IPv6 link-local on Wi-Fi is refused");
    CHECK(allow(SA(lo6), SA(lo6), UNK), "IPv6 loopback is allowed");
    CHECK(!allow(SA(g6), SA(g6), WIRED), "global IPv6 is refused");
    CHECK(!allow(SA(l6), SA(g6), WIRED), "a global IPv6 peer is refused");
    CHECK(allow(SA(m4l), SA(m4p), WIRED), "v4-mapped link-local is judged as IPv4");
    CHECK(!allow(SA(m4l), SA(m4lan), WIRED), "v4-mapped routable peer is refused");
    CHECK(!allow(SA(ll1), SA(l6), WIRED), "mismatched families are refused");
    CHECK(!allow(nullptr, SA(ll1), WIRED), "missing address is refused");

    CHECK(allow(SA(ll1), SA(ll2), 0, "2,0"), "BB_CABLE_IF_TYPES=2,0 admits an UNKNOWN(0)-typed cable");
    CHECK(!allow(SA(ll1), SA(ll2), WIFI, "2,0"), "the escape hatch still refuses Wi-Fi");
    CHECK(!allow(SA(ll1), SA(ll2), WIRED, "0"), "BB_CABLE_IF_TYPES=0 replaces the default");
    CHECK(!allow(SA(lan1), SA(lan2), 0, "0,1,2,3,4,5"), "no type list admits a routable address");
}

static void test_paths() {
    CHECK(bb::safe_doc_path("tail.gguf"), "plain file");
    CHECK(bb::safe_doc_path("anekv/tmpl16k.mlmodelc/weights/weight.bin"), "nested model file");
    CHECK(bb::safe_doc_path("metal/ggml-metal-embed-fa.metal"), "kernel override");
    CHECK(bb::safe_doc_path("env.txt"), "env.txt");
    for (const char * bad : { "", "/etc/passwd", "../Library/Preferences/x.plist", "a/../../b", "a/./b", "./a", "a//b", "a/", ".hidden",
                              "a/.git/config", "a b", "a\\b", "a;b", "%2e%2e/x", "~/x", "a\nb" }) {
        CHECK(!bb::safe_doc_path(bad), bad);
    }
    CHECK(!bb::safe_doc_path(std::string(600, 'a')), "overlong path");
    CHECK(!bb::safe_doc_path(std::string("a\0b", 3)), "embedded NUL");

    CHECK(bb::fetch_url_ok("http://169.254.3.4:51234/Models/tail.gguf"), "the Mac on the cable");  // sensitive-ok: test fixtures
    CHECK(bb::fetch_url_ok("http://169.254.3.4/x"), "no port");  // sensitive-ok: test fixtures
    for (const char * bad : { "https://169.254.3.4/x", "http://192.168.1.2:80/x", "http://example.com/x", "http://169.254.3.4.evil.com/x",  // sensitive-ok: test fixtures
                              "http://user@169.254.3.4/x", "http://169.254.3.4@8.8.8.8/x", "http://169.254.3.4:80@8.8.8.8/x", "file:///etc/passwd",  // sensitive-ok: test fixtures
                              "http://[fe80::1]/x", "http://169.254.3/x", "http://0xa9fe0304/x", "http://169.254.3.4:/x", "http://169.254.3.4:8a/x",  // sensitive-ok: test fixtures
                              "HTTP://169.254.3.4/x", "http://127.0.0.1/x", "" }) {  // sensitive-ok: test fixtures
        CHECK(!bb::fetch_url_ok(bad), bad);
    }
}

// a real listener on 127.0.0.1 and on this Mac's routable address: cable_accept() and the servers' default filter
static int listen_on(const sockaddr_in & a) {
    int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(s, (const sockaddr *) &a, sizeof a) != 0 || listen(s, 1) != 0) { close(s); return -1; }
    return s;
}
static int port_of(int s) { sockaddr_in a = {}; socklen_t l = sizeof a; getsockname(s, (sockaddr *) &a, &l); return ntohs(a.sin_port); }
static std::string routable_v4() {
    ifaddrs * ifs = nullptr; std::string r;
    if (getifaddrs(&ifs) != 0) return r;
    for (ifaddrs * p = ifs; p && r.empty(); p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        const uint32_t a = ntohl(((sockaddr_in *) p->ifa_addr)->sin_addr.s_addr);
        if (bb::v4_loopback(a) || bb::v4_link_local(a)) continue;
        char b[INET_ADDRSTRLEN]; inet_ntop(AF_INET, &((sockaddr_in *) p->ifa_addr)->sin_addr, b, sizeof b); r = b;
    }
    freeifaddrs(ifs);
    return r;
}
static void live(const char * ip, bool want_cable, bool want_default, const char * what, bool required) {
    int s = listen_on(v4(ip));
    if (s < 0) { fprintf(stderr, "skip live %s (%s): bind failed\n", what, ip); return; }
    std::thread t([&] { int c = socket(AF_INET, SOCK_STREAM, 0); auto a = v4(ip, port_of(s)); connect(c, (sockaddr *) &a, sizeof a); usleep(200000); close(c); });
    pollfd pf = { s, POLLIN, 0 };
    if (poll(&pf, 1, 3000) != 1) {   // a sandbox that blocks sockets: report it rather than hang
        // a firewall or sandbox can block connecting to this Mac's own LAN address: then only the decide() cases above cover
        // routable peers, and scripts/check-phone-exposure.sh covers real Wi-Fi on the phone
        fprintf(stderr, "%s live %s (%s): no connection arrived in 3 s (sockets blocked here)\n", required ? "FAIL" : "skip", what, ip);
        if (required) g_fail++;
        t.join(); close(s); return;
    }
    int fd = accept(s, nullptr, nullptr);
    std::string why;
    CHECK(bb::cable_accept(fd, why) == want_cable, (std::string("cable_accept ") + what + ": " + why).c_str());
    CHECK(pa::accept_loopback_only(fd, why) == want_default, (std::string("phone-attn default filter ") + what + ": " + why).c_str());
    CHECK(spt::accept_loopback_only(fd, why) == want_default, (std::string("tail-server default filter ") + what + ": " + why).c_str());
    close(fd); t.join(); close(s);
}

int main() {
    test_decide();
    test_paths();
    unsetenv("PA_ALLOW_REMOTE"); unsetenv("SPT_ALLOW_REMOTE"); unsetenv("BB_CABLE_IF_TYPES");
    live("127.0.0.1", true, true, "loopback", true);
    const std::string r = routable_v4();
    if (!r.empty()) live(r.c_str(), false, false, "this Mac's LAN address (like a Wi-Fi peer)", false);
    else fprintf(stderr, "skip live LAN test: no routable IPv4\n");
    CHECK(bb::stats().allowed.load() >= 1, "the gate counts allowed connections");
    printf("cable-policy-test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
