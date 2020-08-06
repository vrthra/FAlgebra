from Parser import EarleyParser as Parser
import sys
import json

def main(input_file, grammar_file):
    with open(grammar_file) as f:
        meta_g = json.loads(f.read())
    grammar = meta_g['[grammar]']
    start = meta_g['[start]']
    p = Parser(grammar, start_symbol=start, canonical=True)
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
    print('success:', (success_count_total - success_count_neg),  '/', success_count_total)
    print('fail:', (fail_count_total - fail_count_neg),  '/', fail_count_total)


main(*sys.argv[1:])
