from Parser import EarleyParser as Parser
import sys
import json
import os

def main(grammar, bug_fn, predicate):
    input_file = './results/%s.log.json' % os.path.basename(bug_fn)
    one_fault_grammar_file = './results/%s_atleast_one_fault_g.json' % os.path.basename(bug_fn)

    with open(one_fault_grammar_file) as f:
        one_fault_meta_g = json.loads(f.read())
    one_fault_grammar = one_fault_meta_g['[grammar]']
    one_fault_start = one_fault_meta_g['[start]']
    p = Parser(one_fault_grammar, start_symbol=one_fault_start, canonical=True)
    success_count_total = 0
    success_count_neg = 0
    fail_count_total = 0
    fail_count_neg = 0

    with open(input_file) as f:
        jsoninputs = f.readlines()
    for line in jsoninputs:
        res = json.loads(line)
        if res['res'] == 'PRes.invalid':
            continue   
        elif res['res'] == 'PRes.success':
            success_count_total += 1
            if not p.can_parse(res['src']):
                success_count_neg += 1
                print('ERROR:', res)
        elif res['res'] == 'PRes.failed':
            fail_count_total += 1
            if p.can_parse(res['src']):
                fail_count_neg += 1
        else:
            assert False
    print('Success: %d/%d' % ((success_count_total - success_count_neg), success_count_total))
    print('Fail: %d/%d' % ((fail_count_total - fail_count_neg), fail_count_total))
