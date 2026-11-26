#include "jit_x64.h"

namespace Teakra {

namespace {

void SetFlags(RegisterState& state, Flags flags) {
    state.fr = flags.fr;
    state.flm = flags.flm;
    state.fvl = flags.fvl;
    state.fe = flags.fe;
    state.fc0 = flags.fc0;
    state.fv = flags.fv;
    state.fn = flags.fn;
    state.fm = flags.fm;
    state.fz = flags.fz;
    state.fc1 = flags.fc1;
}

Flags GetFlags(const RegisterState& state) {
    Flags flags{};
    flags.fr.Assign(state.fr);
    flags.flm.Assign(state.flm);
    flags.fvl.Assign(state.fvl);
    flags.fe.Assign(state.fe);
    flags.fc0.Assign(state.fc0);
    flags.fv.Assign(state.fv);
    flags.fn.Assign(state.fn);
    flags.fm.Assign(state.fm);
    flags.fz.Assign(state.fz);
    flags.fc1.Assign(state.fc1);
    return flags;
}

// The registers exchanged by cntx_s/cntx_r, which RegisterState keeps in its shadow lists. The
// user output pins in mod0 are not exchanged.
void SetSwapRegisters(RegisterState& state, u16 pcmhi, Mod0 mod0, Mod1 mod1, Mod2 mod2,
                      const std::array<u16, 3>& im, u16 imv, const std::array<ArU, 2>& ar,
                      const std::array<ArpU, 4>& arp, bool keep_ou) {
    const auto ou = state.ou;
    state.pcmhi = pcmhi;
    state.Set<Teakra::mod0>(mod0.raw);
    state.Set<Teakra::mod1>(mod1.raw);
    state.Set<Teakra::mod2>(mod2.raw);
    state.im = im;
    state.imv = imv;
    state.Set<ar0>(ar[0].raw);
    state.Set<ar1>(ar[1].raw);
    state.Set<arp0>(arp[0].raw);
    state.Set<arp1>(arp[1].raw);
    state.Set<arp2>(arp[2].raw);
    state.Set<arp3>(arp[3].raw);
    if (keep_ou) {
        state.ou = ou;
    }
}

} // Anonymous namespace

void SaveJitRegisters(const JitRegisters& jit, RegisterState& state) {
    // Shadows first, because they are set by writing the primary registers and swapping them.
    SetFlags(state, jit.flagsb);
    state.ShadowStore();
    SetSwapRegisters(state, jit.pcmhib, jit.mod0b, jit.mod1b, jit.mod2b, jit.imb, jit.imvb,
                     jit.arb, jit.arpb, true);
    state.ShadowSwap();

    state.pc = jit.pc;
    state.prpage = jit.prpage;
    state.cpc = jit.cpc;
    state.repc = jit.repc;
    state.repcs = jit.repcs;
    state.rep = jit.rep;
    state.crep = jit.crep;
    state.bcn = jit.bcn;
    state.lp = jit.lp;
    for (std::size_t i = 0; i < state.bkrep_stack.size(); ++i) {
        state.bkrep_stack[i].start = jit.bkrep_stack[i].start;
        state.bkrep_stack[i].end = jit.bkrep_stack[i].end;
        state.bkrep_stack[i].lc = jit.bkrep_stack[i].lc;
    }

    state.a = jit.a;
    state.b = jit.b;
    state.a1s = jit.a1s;
    state.b1s = jit.b1s;
    state.ccnta = jit.ccnta;
    state.sv = jit.sv;
    SetFlags(state, jit.flags);
    state.vtr0 = jit.vtr0;
    state.vtr1 = jit.vtr1;

    state.x = jit.x;
    state.y = jit.y;
    state.p = jit.p;
    state.pe = jit.pe;
    state.p0h_cbs = jit.p0h_cbs;

    state.r = jit.r;
    state.mixp = jit.mixp;
    state.sp = jit.sp;
    state.r0b = jit.r0b;
    state.r1b = jit.r1b;
    state.r4b = jit.r4b;
    state.r7b = jit.r7b;

    state.Set<cfgi>(jit.cfgi.raw);
    state.Set<cfgj>(jit.cfgj.raw);
    state.stepi0 = jit.stepi0;
    state.stepj0 = jit.stepj0;
    state.stepib = jit.cfgib.step;
    state.modib = jit.cfgib.mod;
    state.stepjb = jit.cfgjb.step;
    state.modjb = jit.cfgjb.mod;
    state.stepi0b = jit.stepi0b;
    state.stepj0b = jit.stepj0b;

    state.ip = jit.ip;
    state.ipv = jit.ipv;
    state.ic = jit.ic;
    state.nimc = jit.nimc;
    state.ie = jit.ie;
    state.ou = jit.ou;
    state.iu = jit.iu;
    state.ext = jit.ext;

    SetSwapRegisters(state, jit.pcmhi, jit.mod0, jit.mod1, jit.mod2, jit.im, jit.imv, jit.ar,
                     jit.arp, false);
}

void LoadJitRegisters(JitRegisters& jit, const RegisterState& state) {
    RegisterState shadow = state;
    shadow.ShadowRestore();
    jit.flagsb = GetFlags(shadow);

    shadow = state;
    shadow.ShadowSwap();
    jit.pcmhib = shadow.pcmhi;
    jit.mod0b.raw = shadow.Get<mod0>();
    jit.mod1b.raw = shadow.Get<mod1>();
    jit.mod2b.raw = shadow.Get<mod2>();
    jit.imb = shadow.im;
    jit.imvb = shadow.imv;
    jit.arb[0].raw = shadow.Get<ar0>();
    jit.arb[1].raw = shadow.Get<ar1>();
    jit.arpb[0].raw = shadow.Get<arp0>();
    jit.arpb[1].raw = shadow.Get<arp1>();
    jit.arpb[2].raw = shadow.Get<arp2>();
    jit.arpb[3].raw = shadow.Get<arp3>();

    jit.pc = state.pc;
    jit.prpage = state.prpage;
    jit.cpc = state.cpc;
    jit.repc = state.repc;
    jit.repcs = state.repcs;
    jit.rep = state.rep;
    jit.crep = state.crep;
    jit.bcn = state.bcn;
    jit.lp = state.lp;
    for (std::size_t i = 0; i < state.bkrep_stack.size(); ++i) {
        jit.bkrep_stack[i].start = state.bkrep_stack[i].start;
        jit.bkrep_stack[i].end = state.bkrep_stack[i].end;
        jit.bkrep_stack[i].lc = state.bkrep_stack[i].lc;
    }

    jit.a = state.a;
    jit.b = state.b;
    jit.a1s = state.a1s;
    jit.b1s = state.b1s;
    jit.ccnta = state.ccnta;
    jit.sv = state.sv;
    jit.flags = GetFlags(state);
    jit.vtr0 = state.vtr0;
    jit.vtr1 = state.vtr1;

    jit.x = state.x;
    jit.y = state.y;
    jit.p = state.p;
    jit.pe = state.pe;
    jit.p0h_cbs = state.p0h_cbs;

    jit.r = state.r;
    jit.mixp = state.mixp;
    jit.sp = state.sp;
    jit.r0b = state.r0b;
    jit.r1b = state.r1b;
    jit.r4b = state.r4b;
    jit.r7b = state.r7b;

    jit.cfgi.raw = state.Get<cfgi>();
    jit.cfgj.raw = state.Get<cfgj>();
    jit.stepi0 = state.stepi0;
    jit.stepj0 = state.stepj0;
    jit.cfgib.step.Assign(state.stepib);
    jit.cfgib.mod.Assign(state.modib);
    jit.cfgjb.step.Assign(state.stepjb);
    jit.cfgjb.mod.Assign(state.modjb);
    jit.stepi0b = state.stepi0b;
    jit.stepj0b = state.stepj0b;

    jit.pcmhi = state.pcmhi;
    jit.mod0.raw = state.Get<mod0>();
    jit.mod1.raw = state.Get<mod1>();
    jit.mod2.raw = state.Get<mod2>();
    jit.im = state.im;
    jit.imv = state.imv;
    jit.ar[0].raw = state.Get<ar0>();
    jit.ar[1].raw = state.Get<ar1>();
    jit.arp[0].raw = state.Get<arp0>();
    jit.arp[1].raw = state.Get<arp1>();
    jit.arp[2].raw = state.Get<arp2>();
    jit.arp[3].raw = state.Get<arp3>();

    jit.ip = state.ip;
    jit.ipv = state.ipv;
    jit.ic = state.ic;
    jit.nimc = state.nimc;
    jit.ie = state.ie;
    jit.ou = state.ou;
    jit.iu = state.iu;
    jit.ext = state.ext;
}

} // namespace Teakra
