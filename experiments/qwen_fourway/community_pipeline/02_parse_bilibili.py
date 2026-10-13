import argparse
from common.core import parse_export, write_new

def main():
 p=argparse.ArgumentParser(description='Parse authorized offline Bilibili JSON/JSONL export')
 p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--stats',required=True); p.add_argument('--synthetic-fixture',action='store_true')
 a=p.parse_args(); rows,stats=parse_export(a.input,'bilibili',synthetic=a.synthetic_fixture)
 write_new(a.output,rows,jsonl=True); write_new(a.stats,stats)
 print(stats['retained'],'records;',stats['filtered'],'filtered')
if __name__=='__main__': main()
