#!/usr/bin/env python3
"""CTAF-KHAF Synthetic Dataset Generator: builds 100 scenarios with METAR, ADS-B, transcripts, and TTS audio."""

import os, json, math, random, re, time, argparse, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List

KHAF_LAT     = 37.5134
KHAF_LON     = -122.5008
KHAF_ELEV    = 66
RWY30        = 300
RWY12        = 120
PAT_ALT      = 1066
PAT_OFF      = 0.55
APPROACH_SPD = 90
CRUISE_SPD   = 110

FINAL_ALT = {10:3000, 8:2500, 5:1800, 3:1100, 2:800,
             1.5:600, 1:450, 0.5:250, 0.3:150, 0:KHAF_ELEV}


def move(lat, lon, brg, d):
    R = 3440.065
    b = math.radians(brg); dn = d/R
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1)*math.cos(dn)+math.cos(la1)*math.sin(dn)*math.cos(b))
    lo2 = lo1+math.atan2(math.sin(b)*math.sin(dn)*math.cos(la1),
                          math.cos(dn)-math.sin(la1)*math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)

def dnm(la1,lo1,la2,lo2):
    R=3440.065; dlat=math.radians(la2-la1); dlon=math.radians(lo2-lo1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlon/2)**2
    return R*2*math.asin(math.sqrt(max(0,a)))

def brg(la1,lo1,la2,lo2):
    dlon=math.radians(lo2-lo1)
    x=math.sin(dlon)*math.cos(math.radians(la2))
    y=math.cos(math.radians(la1))*math.sin(math.radians(la2))-math.sin(math.radians(la1))*math.cos(math.radians(la2))*math.cos(dlon)
    return (math.degrees(math.atan2(x,y))+360)%360

def final_alt(d):
    keys=sorted(FINAL_ALT.keys(),reverse=True)
    for i in range(len(keys)-1):
        a,b=keys[i],keys[i+1]
        if b<=d<=a:
            f=(d-b)/(a-b); return FINAL_ALT[b]+f*(FINAL_ALT[a]-FINAL_ALT[b])
    return FINAL_ALT[keys[0]] if d>=keys[0] else KHAF_ELEV


def dw(along=0.3):
    """Right downwind position, along NM past threshold."""
    return move(*move(KHAF_LAT,KHAF_LON,30,PAT_OFF), RWY12, along)

def base(frac=0.5):
    s=dw(1.0); e=move(KHAF_LAT,KHAF_LON,RWY12,0.5)
    return (s[0]+frac*(e[0]-s[0]), s[1]+frac*(e[1]-s[1]))

def upwind(d=0.5):   return move(KHAF_LAT,KHAF_LON,RWY30,d)
def crosswind():      return move(*upwind(0.8),30,PAT_OFF)
def thr30():          return move(KHAF_LAT,KHAF_LON,RWY30,0.35)
def fp(d):            return move(KHAF_LAT,KHAF_LON,RWY12,d)


@dataclass
class Pt:
    t:      float
    lat:    float
    lon:    float
    alt:    float
    hdg:    float
    spd:    float
    phase:  str  = ''
    on_gnd: bool = False


class Aircraft:
    def __init__(self, callsign, ac_type, has_radio, lat, lon, alt, hdg=RWY30):
        self.callsign  = callsign
        self.ac_type   = ac_type
        self.has_radio = has_radio
        self.track: List[Pt] = [Pt(0, lat, lon, alt, hdg, 0)]

    @property
    def cur(self): return self.track[-1]

    def _fly(self, dlat, dlon, dalt, spd, phase, dt=1.0):
        d   = dnm(self.cur.lat,self.cur.lon,dlat,dlon)
        if d < 0.005: return
        h   = brg(self.cur.lat,self.cur.lon,dlat,dlon)
        dur = d/(spd/3600)
        da  = dalt - self.cur.alt
        start_t = self.cur.t          # capture once — self.cur changes as we append
        start_lat, start_lon = self.cur.lat, self.cur.lon
        start_alt = self.cur.alt
        t   = start_t
        while t < start_t + dur - dt/2:
            frac = (t - start_t) / dur
            la,lo = move(start_lat, start_lon, h, d*frac)
            self.track.append(Pt(t,la,lo,start_alt+da*frac,h,spd,phase))
            t += dt
        self.track.append(Pt(start_t+dur,dlat,dlon,dalt,h,0,phase,
                             on_gnd=(dalt<=KHAF_ELEV+30)))

    def fly_to(self, pos, alt, spd=APPROACH_SPD, phase=''):
        self._fly(pos[0],pos[1],alt,spd,phase)

    def hold(self, dur, phase=''):
        c = self.cur
        for i in range(int(dur)):
            self.track.append(Pt(c.t+i+1,c.lat,c.lon,c.alt,c.hdg,0,phase,c.on_gnd))

    def fly_final(self, start_nm, go_around=False):
        d = start_nm
        while d > 0.05:
            nd = max(d-0.35, 0.0)
            self._fly(*fp(nd), final_alt(nd), APPROACH_SPD, f'final_{nd:.1f}')
            d = nd
        if go_around:
            self._fly(*upwind(0.4), 400, APPROACH_SPD, 'go_around')
        else:
            t = thr30()
            self._fly(t[0],t[1],KHAF_ELEV,APPROACH_SPD,'landing')
            self.track[-1].on_gnd = True
            self.hold(15,'ground')

    def fly_pattern(self, entry='downwind_mid', land=True, goaround=False):
        if entry in ('crosswind','downwind_entry','downwind_mid','downwind_base'):
            if entry == 'crosswind':
                self.fly_to(crosswind(),900,APPROACH_SPD,'crosswind')
            if entry in ('crosswind','downwind_entry'):
                self.fly_to(dw(0.0),PAT_ALT,APPROACH_SPD,'downwind_entry')
            if entry in ('crosswind','downwind_entry','downwind_mid'):
                self.fly_to(dw(0.35),PAT_ALT,APPROACH_SPD,'downwind_mid')
            self.fly_to(dw(0.8), PAT_ALT,APPROACH_SPD,'downwind_base')
        self.fly_to(dw(1.0),900,APPROACH_SPD,'base_turn')
        self.fly_to(base(0.5),850,APPROACH_SPD,'base_mid')
        self.fly_to(fp(1.5),600,APPROACH_SPD,'final_1.5')
        self.fly_to(fp(0.5),250,APPROACH_SPD,'final_0.5')
        self.fly_to(fp(0.3),150,APPROACH_SPD,'short_final')
        if goaround:
            self.fly_to(upwind(0.4),400,APPROACH_SPD,'go_around')
            self.fly_to(upwind(0.8),600,APPROACH_SPD,'upwind')
            self.fly_to(crosswind(),900,APPROACH_SPD,'crosswind')
            self.fly_to(dw(0.35),PAT_ALT,APPROACH_SPD,'downwind_mid')
        elif land:
            t=thr30()
            self.fly_to(t,KHAF_ELEV,APPROACH_SPD,'landing')
            self.track[-1].on_gnd=True
            self.hold(15,'ground')

    def as_track(self, interval=10):
        if not self.track: return []
        max_t = int(self.cur.t)
        by_t  = {}
        for p in self.track:
            by_t[int(p.t)] = p
        result = []
        for t in range(0, max_t+interval, interval):
            candidates = [k for k in by_t if k<=t]
            if not candidates: candidates = list(by_t.keys())
            p = by_t[min(candidates, key=lambda x: abs(x-t))]
            result.append({
                'time_s': t, 'lat': round(p.lat,5), 'lon': round(p.lon,5),
                'alt_ft': int(p.alt), 'hdg': int(p.hdg)%360,
                'spd_kt': int(p.spd), 'callsign': self.callsign,
                'aircraft_type': self.ac_type, 'has_radio': self.has_radio,
                'on_ground': p.on_gnd, 'phase': p.phase,
            })
        return result


def rng(sc_id): return random.Random(int(hashlib.md5(sc_id.encode()).hexdigest()[:8],16))

def make(info, lat, lon, alt, hdg=RWY30):
    return Aircraft(info[0],info[1],info[2],lat,lon,alt,hdg)

def sim_simultaneous_final(sc_id, A_info, B_info):
    r = rng(sc_id)
    sa = r.uniform(2.5,4.0)
    A = make(A_info,*fp(sa),final_alt(sa))
    A.fly_final(sa, go_around=True)
    A.fly_to(upwind(0.8),600,phase='upwind')
    A.fly_to(crosswind(),1000,phase='crosswind')
    B = make(B_info,*dw(r.uniform(0.5,0.9)),PAT_ALT)
    B.fly_pattern(entry='downwind_base',land=True)
    return A, B

def sim_runway_incursion(sc_id, A_info, B_info):
    r = rng(sc_id)
    sa = r.uniform(0.4,0.8)
    A = make(A_info,*fp(sa),final_alt(sa))
    A.fly_final(sa, go_around=True)
    A.fly_to(upwind(0.6),600,phase='upwind')
    A.fly_to(crosswind(),900,phase='crosswind')
    A.fly_to(dw(0.35),PAT_ALT,phase='downwind_mid')
    t=thr30()
    B = make(B_info,t[0],t[1],KHAF_ELEV)
    B.track[-1].on_gnd=True
    B.hold(int(A.cur.t)+20,'on_runway')
    return A, B

def sim_go_around_conflict(sc_id, A_info, B_info):
    r = rng(sc_id)
    sa = r.uniform(0.3,0.6)
    A = make(A_info,*fp(sa),final_alt(sa))
    A.fly_final(sa, go_around=True)
    A.fly_to(upwind(0.8),700,phase='upwind')
    A.fly_to(crosswind(),950,phase='crosswind')
    t=thr30()
    B = make(B_info,t[0],t[1],KHAF_ELEV)
    B.track[-1].on_gnd=True
    B.hold(r.uniform(3,8),'holding_short')
    B.fly_to(upwind(0.5),350,phase='upwind_low')
    B.fly_to(upwind(1.2),1000,phase='upwind')
    B.fly_to(crosswind(),1200,phase='crosswind')
    return A, B

def sim_imc_vfr(sc_id, A_info, B_info):
    r = rng(sc_id)
    sa = r.uniform(3.0,5.0)
    A = make(A_info,*fp(sa),final_alt(sa))
    A.fly_final(sa, go_around=True)
    A.fly_to(upwind(1.0),800,phase='upwind')
    A.fly_to(move(KHAF_LAT,KHAF_LON,RWY12,3.0),2000,CRUISE_SPD,'missed_approach')
    sb = r.uniform(4.0,6.0)
    B = make(B_info,*fp(sb),r.uniform(1200,1800))
    imc = move(KHAF_LAT,KHAF_LON,RWY12,1.5)
    B.fly_to(imc, r.uniform(350,500), phase='imc_low')
    B.hold(r.uniform(8,15),'imc_disoriented')
    B.fly_to(move(KHAF_LAT,KHAF_LON,RWY12,4.0),2500,CRUISE_SPD,'imc_climbing')
    return A, B

def sim_silent_traffic(sc_id, A_info, B_info):
    r = rng(sc_id)
    A = make(A_info,*dw(0.3),PAT_ALT)
    A.fly_pattern(entry='downwind_mid',land=False,goaround=True)
    B = make(B_info,*dw(r.uniform(0.4,0.7)),PAT_ALT)
    B.fly_pattern(entry='downwind_mid',land=True)
    return A, B

def sim_wrong_runway(sc_id, A_info, B_info):
    r = rng(sc_id)
    sa = r.uniform(2.0,4.0)
    A = make(A_info,*fp(sa),final_alt(sa))
    A.fly_final(sa,go_around=False)
    sb = r.uniform(2.0,4.0)
    wrong_start = move(KHAF_LAT,KHAF_LON,RWY30,sb)
    B = make(B_info,wrong_start[0],wrong_start[1],final_alt(sb),hdg=RWY12)
    mid = move(KHAF_LAT,KHAF_LON,RWY30,sb/2)
    B.fly_to(mid,final_alt(sb/2),APPROACH_SPD,'wrong_rwy')
    B.fly_to(upwind(0.5),500,APPROACH_SPD,'correcting')
    B.fly_to(crosswind(),PAT_ALT,phase='crosswind')
    return A, B

def sim_pattern_conflict(sc_id, A_info, B_info):
    r = rng(sc_id)
    A = make(A_info,*dw(0.8),PAT_ALT)
    A.fly_pattern(entry='downwind_base',land=True)
    sb = r.uniform(5.0,8.0)
    B = make(B_info,*fp(sb),final_alt(sb))
    B.fly_final(sb,go_around=False)
    return A, B

def sim_missing_position(sc_id, A_info, B_info):
    r = rng(sc_id)
    A = make(A_info,*crosswind(),900)
    A.fly_to(dw(0.35),PAT_ALT,phase='downwind_mid')
    A.hold(r.uniform(15,25),'downwind_mid')
    A.fly_to(dw(1.0),900,phase='base_turn')
    A.fly_final(1.5,go_around=False)
    sb = r.uniform(3.0,5.0)
    B = make(B_info,*fp(sb),final_alt(sb))
    B.fly_final(sb,go_around=False)
    return A, B

def sim_improper_entry(sc_id, A_info, B_info):
    r = rng(sc_id)
    sa = r.uniform(6.0,10.0)
    A = make(A_info,*fp(sa),final_alt(sa))
    A.fly_final(sa,go_around=False)
    B = make(B_info,*dw(0.6),PAT_ALT)
    B.fly_to(dw(1.0),900,phase='downwind_base')
    B.hold(r.uniform(10,20),'extended_downwind')
    B.fly_final(1.5,go_around=False)
    return A, B

def sim_nominal_single(sc_id, A_info, B_info):
    r = rng(sc_id)
    A = make(A_info,*dw(0.0),PAT_ALT)
    A.fly_pattern(entry='downwind_entry',land=True)
    B = make(B_info,KHAF_LAT,KHAF_LON,KHAF_ELEV)
    B.track[-1].on_gnd=True
    B.hold(int(A.cur.t)+10,'parked')
    return A, B

def sim_nominal_multi(sc_id, A_info, B_info):
    r = rng(sc_id)
    A = make(A_info,*dw(r.uniform(0.1,0.4)),PAT_ALT)
    A.fly_pattern(entry='downwind_mid',land=True)
    B = make(B_info,*crosswind(),900)
    B.fly_to(dw(0.0),PAT_ALT,phase='downwind_entry')
    B.fly_to(dw(0.35),PAT_ALT,phase='downwind_mid')
    B.hold(r.uniform(10,20),'extended_downwind')
    B.fly_to(dw(1.0),900,phase='base_turn')
    B.fly_final(1.5,go_around=False)
    return A, B

def sim_nominal_instrument(sc_id, A_info, B_info):
    r = rng(sc_id)
    sa = r.uniform(5.0,8.0)
    A = make(A_info,*fp(sa),final_alt(sa))
    A.fly_final(sa,go_around=False)
    B = make(B_info,*dw(0.35),PAT_ALT)
    B.hold(r.uniform(15,25),'extended_downwind')
    B.fly_to(dw(1.0),900,phase='base_turn')
    B.fly_final(1.5,go_around=False)
    return A, B

SIMULATORS = {
    'simultaneous_final':         sim_simultaneous_final,
    'runway_incursion_risk':       sim_runway_incursion,
    'go_around_conflict':          sim_go_around_conflict,
    'imc_vfr_conflict':            sim_imc_vfr,
    'silent_traffic':              sim_silent_traffic,
    'wrong_runway_announcement':   sim_wrong_runway,
    'pattern_conflict':            sim_pattern_conflict,
    'missing_position_calls':      sim_missing_position,
    'improper_entry':              sim_improper_entry,
    'nominal_single_aircraft':     sim_nominal_single,
    'nominal_multi_aircraft':      sim_nominal_multi,
    'nominal_instrument_approach': sim_nominal_instrument,
}


SCENARIO_PLAN = [
    ('simultaneous_final',         'hazard',   9),
    ('runway_incursion_risk',       'hazard',   8),
    ('go_around_conflict',          'hazard',   7),
    ('imc_vfr_conflict',            'hazard',   8),  # hazard: 32 total — close enough
    ('silent_traffic',              'warning',  7),
    ('wrong_runway_announcement',   'warning',  8),
    ('pattern_conflict',            'warning',  7),
    ('missing_position_calls',      'warning',  7),
    ('improper_entry',              'warning',  6),  # warning: 35 total
    ('nominal_single_aircraft',     'nominal',  11),
    ('nominal_multi_aircraft',      'nominal',  11),
    ('nominal_instrument_approach', 'nominal',  11),  # nominal: 33 total
]

CALLSIGNS = ['N910YZ','N602SK','N778EH','N75WW','N034CR','N881TF',
             'N321FG','N405CC','N412PB','N556KA','N8AC','N689PQ',
             'N203JL','N134MV','N990DX','N753HN','N351KT','N247GS','N223BT','N567RA']

AC_TYPES = ['Cessna 172','Cessna 182','Piper Archer','Piper Cherokee',
            'Piper Seneca','Beechcraft Bonanza','Mooney M20','Cirrus SR22',
            'Diamond DA40','Cessna 210']

METAR_POOL = [
    ('KHAF {d}Z AUTO {wd:03d}{wk:02d}KT 10SM CLR {t:02d}/{dp:02d} A{alt} RMK AO2',
     'VFR clear', None, 10),
    ('KHAF {d}Z AUTO {wd:03d}{wk:02d}KT 9SM SCT025 {t:02d}/{dp:02d} A{alt} RMK AO2',
     'VFR scattered 2500ft', 2500, 9),
    ('KHAF {d}Z AUTO {wd:03d}{wk:02d}KT 7SM BKN025 {t:02d}/{dp:02d} A{alt} RMK AO2',
     'MVFR broken 2500ft', 2500, 7),
    ('KHAF {d}Z AUTO {wd:03d}{wk:02d}KT 5SM -BR FEW010 BKN020 {t:02d}/{dp:02d} A{alt} RMK AO2',
     'MVFR mist ceiling 2000ft', 2000, 5),
    ('KHAF {d}Z AUTO {wd:03d}{wk:02d}KT 10SM OVC016 {t:02d}/{dp:02d} A{alt} RMK AO2',
     'Overcast 1600ft', 1600, 10),
]

def gen_metar(r):
    tmpl,desc,ceil,vis = r.choice(METAR_POOL)
    wd=r.choice([180,210,270,300,330,360,30]); wk=r.randint(3,18)
    t=r.randint(10,22); dp=t-r.randint(2,6); alt_v=r.randint(2990,3010)
    day=r.randint(1,28); hr=r.randint(14,22); mn=r.choice([15,35,55])
    raw=tmpl.format(d=f'{day:02d}{hr:02d}{mn:02d}',wd=wd,wk=wk,t=t,dp=dp,alt=alt_v)
    return {'raw':raw,'description':desc,'ceiling_ft':ceil,'visibility_sm':vis,
            'wind_direction':wd,'wind_kt':wk,'temperature_c':t,'dewpoint_c':dp}


# Map fine-grained final_X.X phases to human-readable milestones
def _coarsen_phase(phase, dist_nm):
    """Collapse final_X.X into ~5 meaningful position calls."""
    if phase.startswith('final_') or phase in ('landing','ground'):
        if dist_nm >= 7:   return 'final_8nm'
        if dist_nm >= 4.5: return 'final_5nm'
        if dist_nm >= 2.5: return 'final_3nm'
        if dist_nm >= 1.2: return 'final_2nm'
        if dist_nm >= 0.7: return 'final_1nm'
        if dist_nm >= 0.4: return 'final_half_nm'
        if dist_nm >= 0.2: return 'short_final'
        if dist_nm >= 0.05:return 'over_numbers'
        return phase  # landing/ground
    return phase

def extract_events(ac: Aircraft):
    """Extract key position milestones, max ~6 per aircraft."""
    raw=[]; last_phase=''; last_t=-999
    for p in ac.track:
        if p.phase==last_phase and p.t-last_t<8: continue
        if not p.phase or p.phase in ('','parked'): continue
        d=dnm(KHAF_LAT,KHAF_LON,p.lat,p.lon)
        coarse = _coarsen_phase(p.phase, d)
        if coarse==last_phase and p.t-last_t<15: continue
        raw.append({'t':int(p.t),'phase':coarse,'dist_nm':round(d,1),
                    'alt_ft':int(p.alt),'callsign':ac.callsign,
                    'has_radio':ac.has_radio,'on_ground':p.on_gnd})
        last_phase=coarse; last_t=p.t

    # Keep at most 7 events per aircraft — pick evenly spaced ones
    if len(raw) > 7:
        step = len(raw) / 7
        raw = [raw[int(i*step)] for i in range(7)]
    return raw


SYS_TX = """You generate realistic CTAF radio transcripts for Half Moon Bay Airport (KHAF), runway 30, right-traffic pattern.

Given exact aircraft positions, write an SRT-format transcript of pilot radio calls.

FORMAT (strict):
{index}
{HH:MM:SS,mmm} --> {HH:MM:SS,mmm}
{text}

RULES:
- Use NATO phonetic alphabet for letters (Alpha, Bravo...) and "niner" for 9
- Each self-announced call: "Half Moon Bay traffic, [callsign], [position], [runway 30], [intention], Half Moon Bay."
- NORDO aircraft: only mentioned by other pilots ("Half Moon Bay traffic, blind traffic in the pattern...")
- Timing: each utterance 3-6 seconds long; gap between calls 3-8 seconds
- Timestamps start at 00:00:00,000
- CRITICAL: Total scenario duration MUST be under 90 seconds. If the last timestamp would exceed 90s, stop writing calls early.
- CRITICAL: Write at most 10 lines total. Stop at 10 even if not all events are covered.
- Return ONLY raw SRT — NO markdown fences, NO ```srt, NO ``` anywhere in output.
- Cover the KEY position events only — not every single distance update
- Pilots call position at major phase changes: entering downwind, turning base, turning final, short final, going around, clear of runway
- Do NOT have pilots repeat the same position multiple times unless there is a conflict
- For imc/disoriented pilots: write hesitant, confused speech ("uh", "I got...", corrections)
- Return ONLY the SRT content"""

SYS_ADV = """You are an AI aviation safety advisor monitoring CTAF at KHAF (Half Moon Bay Airport).
Write a concise ground-truth safety advisory (2-4 sentences, ~100-200 words) based on the scenario.
Identify aircraft by callsign and type, state their positions precisely, assess the safety situation, and give recommended actions if needed.
Return ONLY the advisory text."""

def tx_prompt(sc_type, label, A, B, evA, evB, metar, dur):
    evts = sorted(evA+evB, key=lambda x: x['t'])
    ev_str = '\n'.join(
        f"  t={e['t']:3d}s  {e['callsign']}  {e['phase']}  "
        f"{e['dist_nm']}NM  {e['alt_ft']}ft  {'NORDO' if not e['has_radio'] else 'radio'}"
        for e in evts)
    return (f"SCENARIO: {sc_type} ({label})\nMETAR: {metar['raw']}\n"
            f"DURATION: ~{dur}s\n\nAIRCRAFT:\n"
            f"  {A.callsign} ({A.ac_type}) — {'radio' if A.has_radio else 'NORDO'}\n"
            f"  {B.callsign} ({B.ac_type}) — {'radio' if B.has_radio else 'NORDO'}\n\n"
            f"POSITION EVENTS:\n{ev_str}\n\nWrite the SRT transcript.")

def adv_prompt(sc_type, label, A, B, evA, evB):
    evts = sorted(evA+evB, key=lambda x: x['t'])[:8]
    ev_str = '\n'.join(f"  t={e['t']}s  {e['callsign']}  {e['phase']}  {e['dist_nm']}NM  {e['alt_ft']}ft" for e in evts)
    return (f"Scenario: {sc_type} ({label})\n"
            f"  {A.callsign} ({A.ac_type}): {evA[0]['phase'] if evA else 'unknown'}\n"
            f"  {B.callsign} ({B.ac_type}): {evB[0]['phase'] if evB else 'unknown'}\n"
            f"Events:\n{ev_str}\nWrite the safety advisory.")

def gpt(client, system, user, model, max_tokens=1200, temp=0.7, retries=3):
    for i in range(retries):
        try:
            print(f"    → calling {model} (attempt {i+1})...", flush=True)
            r = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=temp,
                timeout=60,
                messages=[{'role':'system','content':system},
                           {'role':'user','content':user}])
            txt = r.choices[0].message.content.strip()
            print(f"    ✓ got {len(txt)} chars", flush=True)
            return txt
        except Exception as e:
            print(f"    API error (attempt {i+1}): {e}")
            time.sleep(2**i)
    return None

def parse_srt(raw):
    # Strip markdown code fences GPT sometimes wraps output in
    raw = re.sub(r'^```(?:srt)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```\s*$', '', raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()

    entries=[]; blocks=re.split(r'\n\n+', raw)
    for b in blocks:
        lines=b.strip().splitlines()
        if len(lines)<3: continue
        try:
            idx=int(lines[0].strip()); ts=lines[1].strip()
            text=' '.join(l.strip() for l in lines[2:])
            # Strip any trailing backticks GPT left in the text
            text = re.sub(r'\s*`+\s*$', '', text).strip()
            if not text: continue
            entries.append({'index':idx,'timestamp':ts,'text':text})
        except: continue

    # Renumber from 1 in case GPT started at 0 or 2
    for i, e in enumerate(entries):
        e['index'] = i + 1
    return entries

def ts_sec(ts):
    m=re.match(r'(\d+):(\d+):(\d+)[,.](\d+)',ts)
    return int(m.group(1))*3600+int(m.group(2))*60+int(m.group(3)) if m else 0


VOICES=['alloy','echo','fable','onyx','nova','shimmer']

def gen_audio(client, entries, callsigns, out_dir: Path):
    try:
        from pydub import AudioSegment
    except ImportError:
        return False
    vm={cs:VOICES[i%len(VOICES)] for i,cs in enumerate(callsigns)}
    segs=[]
    for e in entries:
        voice='alloy'
        for cs in callsigns:
            if cs[1:] in e['text'].upper() or cs[-3:] in e['text'].upper():
                voice=vm.get(cs,'alloy'); break
        try:
            resp=client.audio.speech.create(model='tts-1-hd',voice=voice,
                                             input=e['text'],response_format='mp3')
            tmp=out_dir/f'_l{e["index"]}.mp3'
            tmp.write_bytes(resp.content)
            seg=AudioSegment.from_mp3(tmp)
            parts=e['timestamp'].split('-->')
            if len(parts)==2:
                dur=ts_sec(parts[1].strip())-ts_sec(parts[0].strip())
                sil=max(0,(dur-len(seg)/1000+0.3))*1000
                seg=seg+AudioSegment.silent(int(sil))
            segs.append(seg)
        except Exception as ex:
            print(f"    TTS line {e['index']}: {ex}")
    for f in out_dir.glob('_l*.mp3'): f.unlink()
    if not segs: return False
    out=segs[0]
    for s in segs[1:]: out=out+s
    out.export(out_dir/'audio.mp3',format='mp3')
    return True


def assign_aircraft(r, sc_type):
    cs=r.sample(CALLSIGNS,2); tp=r.sample(AC_TYPES,2)
    nordo_B = sc_type in ('silent_traffic','runway_incursion_risk')
    return (cs[0],tp[0],True), (cs[1],tp[1],not nordo_B)

def generate_scenario(sc_id, sc_type, label, client, out_base: Path,
                       model='gpt-4o', skip_audio=False):
    print(f"  [{sc_id}] {sc_type} ({label})", flush=True)
    r=rng(sc_id)
    A_info,B_info=assign_aircraft(r,sc_type)
    metar=gen_metar(r)

    # 1. Simulate
    print(f"    step 1: simulating...", flush=True)
    A,B = SIMULATORS[sc_type](sc_id, A_info, B_info)
    evA,evB = extract_events(A), extract_events(B)
    dur = min(max(int(A.cur.t),int(B.cur.t))+10, 90)

    # 2. Transcript
    print(f"    step 2: generating transcript...", flush=True)
    raw_srt = gpt(client, SYS_TX, tx_prompt(sc_type,label,A,B,evA,evB,metar,dur), model)
    if not raw_srt:
        raw_srt=f"1\n00:00:00,000 --> 00:00:05,000\nHalf Moon Bay traffic, {A.callsign}, in the pattern, Half Moon Bay."
    entries = parse_srt(raw_srt)
    if not entries:
        entries=[{'index':1,'timestamp':'00:00:00,000 --> 00:00:05,000',
                  'text':f'Half Moon Bay traffic, {A.callsign}, in the pattern.'}]

    # 3. Advisory
    print(f"    step 3: generating advisory...", flush=True)
    advisory = gpt(client, SYS_ADV, adv_prompt(sc_type,label,A,B,evA,evB),
                   model, max_tokens=300, temp=0.5)
    if not advisory:
        advisory=f"Operations at KHAF runway 30. {A.callsign} and {B.callsign} in traffic pattern."

    # 4. Audio
    sc_dir=out_base/'scenarios'/sc_id; sc_dir.mkdir(parents=True,exist_ok=True)
    audio_ok=False
    if not skip_audio:
        audio_ok=gen_audio(client,entries,[A.callsign,B.callsign],sc_dir)

    # 5. Build output
    tA,tB=A.as_track(),B.as_track()
    def snap(ac):
        p=ac.track[0] if ac.track else None
        return {'callsign':ac.callsign,'aircraft_type':ac.ac_type,
                'latitude':round(p.lat,5),'longitude':round(p.lon,5),
                'altitude_ft':int(p.alt),'heading':int(p.hdg)%360,
                'velocity_kt':int(p.spd),'has_radio':ac.has_radio,
                'on_ground':p.on_gnd} if p else {}

    sc = {
        'scenario_id': sc_id, 'label': label, 'hazard_type': sc_type,
        'airport': 'KHAF', 'runway': '30', 'traffic_pattern': 'right',
        'metar': metar, 'adsb_states': [snap(A),snap(B)],
        'adsb_trajectories': {A.callsign: tA, B.callsign: tB},
        'aircraft': [{'callsign':A.callsign,'type':A.ac_type},
                      {'callsign':B.callsign,'type':B.ac_type}],
        'transcript_ground_truth': entries,
        'transcript_ground_truth_raw': raw_srt,
        'transcript_whisper': [], 'transcript_whisper_raw': '',
        'audio_file': f'dataset/scenarios/{sc_id}/audio.mp3',
        'missing_calls': [], 'ground_truth_advisory': advisory,
        'num_gt_entries': len(entries), 'num_whisper_entries': 0,
        'audio_generated': audio_ok, 'whisper_transcribed': False,
    }
    (sc_dir/'scenario.json').write_text(json.dumps(sc,indent=2))
    print(f"    ✓  {len(entries)} tx lines  {len(tA)+len(tB)} track pts  audio={'yes' if audio_ok else 'no'}")
    return sc

def build_plan(n=100, seed=42):
    r=random.Random(seed)
    items=[]
    for sc_type,label,count in SCENARIO_PLAN:
        items.extend([(sc_type,label)]*count)
    r.shuffle(items); items=items[:n]
    return [(f'S{i+1:03d}',st,lb) for i,(st,lb) in enumerate(items)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',        default='dataset_v2')
    ap.add_argument('--samples',    nargs='+', default=None,
                    help='Specific IDs to generate, e.g. S001 S002')
    ap.add_argument('--n',          type=int, default=100)
    ap.add_argument('--model',      default='gpt-4o')
    ap.add_argument('--skip-audio', action='store_true')
    ap.add_argument('--force',      action='store_true')
    args=ap.parse_args()

    api_key=os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...")
        raise SystemExit(1)

    from openai import OpenAI
    client=OpenAI(api_key=api_key)

    out_base=Path(args.out); out_base.mkdir(parents=True,exist_ok=True)
    dst=out_base/'ctaf_khaf_v2.json'

    if dst.exists() and not args.force:
        with open(dst) as f: existing=json.load(f)
        done={s['scenario_id'] for s in existing['scenarios']}
        all_sc=existing['scenarios']
        print(f"Resuming — {len(done)} already done")
    else:
        done=set(); all_sc=[]

    plan=build_plan(args.n)
    if args.samples:
        plan=[(sid,st,lb) for sid,st,lb in plan if sid in args.samples]
        if not plan:
            type_cycle=[(st,lb) for _,st,lb in build_plan(100)]
            plan=[(sid,*type_cycle[i%len(type_cycle)]) for i,sid in enumerate(sorted(args.samples))]

    print(f"\nGenerating {len(plan)} scenarios → {out_base}/")
    print(f"Model: {args.model}  Skip audio: {args.skip_audio}\n")

    for sc_id,sc_type,label in plan:
        if sc_id in done and not args.force:
            print(f"  [{sc_id}] cached"); continue
        try:
            sc=generate_scenario(sc_id,sc_type,label,client,out_base,
                                  args.model,args.skip_audio)
            all_sc=[s for s in all_sc if s['scenario_id']!=sc_id]+[sc]
            all_sc.sort(key=lambda s:s['scenario_id'])
        except Exception as e:
            import traceback
            print(f"  [{sc_id}] ERROR: {e}"); traceback.print_exc(); continue

        dataset={'metadata':{'name':'CTAF-KHAF-Synthetic-v2',
                              'description':'Physics-first synthetic CTAF dataset',
                              'airport':'KHAF','airport_lat':KHAF_LAT,'airport_lon':KHAF_LON,
                              'runway':'30','traffic_pattern':'right',
                              'total_scenarios':len(all_sc),
                              'generation_model_transcript':args.model,
                              'generation_model_tts':'tts-1-hd',
                              'reference_paper':'Darrell et al., AIAA 2026'},
                  'label_schema':{'nominal':'Normal operations',
                                   'warning':'Potential conflict, recoverable',
                                   'hazard':'Imminent safety risk'},
                  'pipeline':{'step_1':'Deterministic physics simulation → ADS-B trajectories',
                               'step_2':f'{args.model} writes transcript from position events',
                               'step_3':'OpenAI TTS synthesizes audio'},
                  'scenarios':all_sc}
        dst.write_text(json.dumps(dataset,indent=2))
        time.sleep(0.4)

    ldist={}
    for s in all_sc: ldist[s['label']]=ldist.get(s['label'],0)+1
    print(f"\nDone. {len(all_sc)} scenarios in {dst}")
    print(f"Labels: {ldist}")

if __name__=='__main__':
    main()