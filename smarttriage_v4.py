import csv, json, heapq, sys
from collections import defaultdict

def read_patients(filepath):
    patients = []
    with open(filepath, newline='') as f:
        for row in csv.DictReader(f):
            patients.append({
                'patient_id':              row['patient_id'],
                'severity':                int(row['severity']),
                'arrival_time':            int(row['arrival_time']),
                'treatment_time':          int(row['treatment_time']),
                'required_specialization': row['required_specialization'].strip()
            })
    return patients

def compute_risk(treatments, patients):
    pm = {p['patient_id']: p for p in patients}
    return sum(pm[t['patient_id']]['severity'] * (t['start_time'] - pm[t['patient_id']]['arrival_time'])
               for t in treatments)

def validate(treatments, patients):
    pm = {p['patient_id']: p for p in patients}
    errors, seen = [], set()
    for t in treatments:
        pid = t['patient_id']
        if pid not in pm: errors.append(f"Unknown: {pid}"); continue
        if pid in seen:   errors.append(f"Duplicate: {pid}")
        seen.add(pid)
        p = pm[pid]
        if t['start_time'] < p['arrival_time']:
            errors.append(f"{pid}: starts before arrival")
        if t['end_time'] != t['start_time'] + p['treatment_time']:
            errors.append(f"{pid}: wrong end_time")
        doc, spec = t['doctor_id'], p['required_specialization']
        if doc == 'Doctor_T' and spec != 'TRAUMA':  errors.append(f"{pid}: Doctor_T cant treat {spec}")
        if doc == 'Doctor_C' and spec != 'CARDIO':  errors.append(f"{pid}: Doctor_C cant treat {spec}")
        if doc not in ('Doctor_T','Doctor_C','Doctor_G'): errors.append(f"{pid}: bad doctor")
    for pid in pm:
        if pid not in seen: errors.append(f"Missing: {pid}")
    slots = defaultdict(list)
    for t in treatments:
        slots[t['doctor_id']].append((t['start_time'], t['end_time'], t['patient_id']))
    for doc, times in slots.items():
        times.sort()
        for i in range(len(times)-1):
            if times[i][1] > times[i+1][0]:
                errors.append(f"OVERLAP {doc}: {times[i][2]} and {times[i+1][2]}")
    return errors

# ═══════════════════════════════════════════════════════════════════
#  STRATEGY TOURNAMENT — 5 strategies, best risk wins
#
#  Strategy 1: WSPT       — severity / treatment_time
#  Strategy 2: Aging      — severity × (1 + 0.2 × wait)
#  Strategy 3: WSPT+Aging — hybrid of 1 and 2
#  Strategy 4: Severity   — severity only (baseline)
#  Strategy 5: Conservative Doctor_G  ← NEW
#              Same as V2: Doctor_G only steals TRAUMA/CARDIO
#              when specialist is busy AND won't finish before
#              the overflow patient's treatment ends.
#              This is more careful with Doctor_G and works
#              better when TRAUMA queue is heavy.
#
#  Each strategy is run independently. The one with the
#  lowest total risk score is selected as the final answer.
# ═══════════════════════════════════════════════════════════════════

def make_priority_fn(strategy):
    if strategy == 'wspt':
        return lambda p, t: p['severity'] / p['treatment_time']
    elif strategy == 'aging':
        return lambda p, t: p['severity'] * (1 + 0.2 * max(0, t - p['arrival_time']))
    elif strategy == 'wspt_aging':
        return lambda p, t: (p['severity'] / p['treatment_time']) * (1 + 0.15 * max(0, t - p['arrival_time']))
    else:
        return lambda p, t: float(p['severity'])

def run_strategy(patients, priority_fn, conservative_g=False):
    """
    Run one full simulation with the given priority function.
    conservative_g=True → Doctor_G only steals overflow when
    specialist is busy AND won't free up before treatment ends.
    conservative_g=False → Doctor_G steals any time specialist is busy.
    """
    by_arrival     = sorted(patients, key=lambda p: p['arrival_time'])
    doctor_free_at = {'Doctor_T': 0, 'Doctor_C': 0, 'Doctor_G': 0}
    queues         = {'TRAUMA': [], 'CARDIO': [], 'GENERAL': []}
    scheduled      = set()
    treatments     = []
    arr_idx        = 0

    def enqueue_arrivals(up_to):
        nonlocal arr_idx
        while arr_idx < len(by_arrival) and by_arrival[arr_idx]['arrival_time'] <= up_to:
            p = by_arrival[arr_idx]
            heapq.heappush(queues[p['required_specialization']],
                           (-p['severity'], p['arrival_time'], p['patient_id'], p))
            arr_idx += 1

    def peek(spec, t):
        while queues[spec] and queues[spec][0][2] in scheduled:
            heapq.heappop(queues[spec])
        valid = [e for e in queues[spec] if e[2] not in scheduled]
        if not valid: return None
        return max(valid, key=lambda e: priority_fn(e[3], t))[3]

    def pop_best(spec, t):
        valid = [(priority_fn(e[3], t), e) for e in queues[spec] if e[2] not in scheduled]
        if not valid: return None
        best_entry = max(valid, key=lambda x: x[0])[1]
        scheduled.add(best_entry[2])
        return best_entry[3]

    def best_across(specs, t):
        best_spec, best_p, best_score = None, None, -1e18
        for spec in specs:
            p = peek(spec, t)
            if p:
                score = priority_fn(p, t)
                if score > best_score:
                    best_score, best_spec, best_p = score, spec, p
        return best_spec, best_p

    def try_assign(doc, t):
        if doctor_free_at[doc] > t:
            return False

        if doc == 'Doctor_T':
            spec, p = best_across(['TRAUMA'], t)

        elif doc == 'Doctor_C':
            spec, p = best_across(['CARDIO'], t)

        else:  # Doctor_G
            gen_spec,    gen_p    = best_across(['GENERAL'], t)
            trauma_spec, trauma_p = best_across(['TRAUMA'],  t)
            cardio_spec, cardio_p = best_across(['CARDIO'],  t)

            candidates = []
            if gen_p:
                candidates.append((gen_spec, gen_p))

            if conservative_g:
                # V2 rule: only steal if specialist won't finish
                # before overflow patient's treatment ends
                if trauma_p:
                    specialist_wait = max(0, doctor_free_at['Doctor_T'] - t)
                    if specialist_wait > trauma_p['treatment_time']:
                        candidates.append((trauma_spec, trauma_p))
                if cardio_p:
                    specialist_wait = max(0, doctor_free_at['Doctor_C'] - t)
                    if specialist_wait > cardio_p['treatment_time']:
                        candidates.append((cardio_spec, cardio_p))
            else:
                # Aggressive rule: steal whenever specialist is busy
                if trauma_p and doctor_free_at['Doctor_T'] > t:
                    candidates.append((trauma_spec, trauma_p))
                if cardio_p and doctor_free_at['Doctor_C'] > t:
                    candidates.append((cardio_spec, cardio_p))

            if not candidates:
                return False

            spec, p = max(candidates, key=lambda x: priority_fn(x[1], t))

        if p is None:
            return False

        pop_best(spec, t)
        start = max(t, p['arrival_time'])
        end   = start + p['treatment_time']
        treatments.append({'patient_id': p['patient_id'], 'doctor_id': doc,
                           'start_time': start, 'end_time': end})
        doctor_free_at[doc] = end
        return True

    prev_time = -1
    while len(scheduled) < len(patients):
        candidate_times = set(doctor_free_at.values())
        if arr_idx < len(by_arrival):
            candidate_times.add(by_arrival[arr_idx]['arrival_time'])
        future = {t for t in candidate_times if t > prev_time}
        if not future: break
        current_time = min(future)
        prev_time    = current_time
        enqueue_arrivals(current_time)
        progress = True
        while progress:
            progress = False
            for doc in ('Doctor_T', 'Doctor_C', 'Doctor_G'):
                if try_assign(doc, current_time):
                    progress = True

    treatments.sort(key=lambda x: x['start_time'])
    return treatments

def schedule(patients):
    """Run all 5 strategies, return the one with lowest risk."""
    configs = [
        ('wspt',             make_priority_fn('wspt'),        False),
        ('aging',            make_priority_fn('aging'),       False),
        ('wspt_aging',       make_priority_fn('wspt_aging'),  False),
        ('severity_only',    make_priority_fn('severity'),    False),
        ('conservative_g',   make_priority_fn('wspt'),        True),   # V2 style
    ]
    best_risk, best_treatments, best_name = float('inf'), None, ''

    print("  Strategy Tournament:")
    print(f"  {'Strategy':<20} {'Risk':>8}")
    print(f"  {'-'*20} {'----':>8}")
    for name, fn, cons_g in configs:
        t = run_strategy(patients, fn, conservative_g=cons_g)
        r = compute_risk(t, patients)
        tag = ' ← BEST' if r < best_risk else ''
        print(f"  {name:<20} {r:>8}{tag}")
        if r < best_risk:
            best_risk, best_treatments, best_name = r, t, name

    print(f"\n  🏆 Winner: {best_name}  (Risk: {best_risk})\n")
    return best_treatments

def write_output(treatments, risk, path='submission.json'):
    with open(path, 'w') as f:
        json.dump({'treatments': treatments, 'estimated_total_risk': risk}, f, indent=2)
    print(f"✅  Saved → {path}   |   📊 Total Risk: {risk}")

if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'patients.csv'
    print(f"\n📂  Reading: {inp}")
    patients = read_patients(inp)
    print(f"    Loaded {len(patients)} patients\n")
    print("⚙️   Scheduling...\n")
    treatments = schedule(patients)
    print("🔍  Validating...")
    errors = validate(treatments, patients)
    if errors:
        print("❌  Errors:"); [print(f"    • {e}") for e in errors]
    else:
        risk = compute_risk(treatments, patients)
        print("    All checks passed!\n")
        write_output(treatments, risk)
        print("\n📋  Final Schedule:")
        print(f"    {'Patient':<10} {'Doctor':<12} {'Start':>6} {'End':>6} {'Wait':>6} {'Risk':>6}")
        print(f"    {'-'*10} {'-'*12} {'-----':>6} {'---':>6} {'----':>6} {'----':>6}")
        pm = {p['patient_id']: p for p in patients}
        for t in treatments:
            p    = pm[t['patient_id']]
            wait = t['start_time'] - p['arrival_time']
            print(f"    {t['patient_id']:<10} {t['doctor_id']:<12} {t['start_time']:>6} {t['end_time']:>6} {wait:>6} {p['severity']*wait:>6}")
