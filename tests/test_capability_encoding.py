"""Non-UTF8 locale regression and malformed JSON diagnostics; offline only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.evaluation.records import RunRecord,atomic_json
from evaluation.summarize_capability import collect
from evaluation.run_capability import load_suite

class EncodingTests(unittest.TestCase):
    def test_utf8_results_under_gbk_default(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            atomic_json(root/'b.manifest.json',{'tasks':[{'case_id':'C','scheme':'agent','repetition':1,'turn_count':1}]})
            r=RunRecord(root,{'id':'C','turn_index':1,'question':'心率依据哪些间隔？','rubric':['完整解释']},{'batch_id':'b','scheme':'agent','repetition':1})
            r.finish({'status':'completed_draft','draft':{'answer':'中文心率说明'}},{},{'metrics':{}},wall_seconds=1)
            original=Path.read_text
            def gbk_read(path,encoding=None,errors=None):
                return original(path,encoding=encoding or 'gbk',errors=errors)
            with patch.object(Path,'read_text',gbk_read):
                report,packet=collect(root,'b')
                suite=load_suite(Path(__file__).resolve().parents[1]/'evaluation/capability_cases.jsonl')
            self.assertEqual(report['groups']['agent']['recorded'],1)
            self.assertEqual(packet[0]['answer'],'中文心率说明')
            self.assertEqual(len(suite),48)
            self.assertEqual(report['invalid_records'],[])
    def test_invalid_json_has_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic_json(root/'b.manifest.json',{'tasks':[]})
            p=root/'broken';p.mkdir();(p/'result.json').write_bytes(b'{invalid')
            report,_=collect(root,'b')
            self.assertEqual(report['invalid_record_details'][0]['error_type'],'JSONDecodeError')
            self.assertEqual(report['invalid_record_details'][0]['line'],1)
    def test_invalid_utf8_has_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic_json(root/'b.manifest.json',{'tasks':[]})
            p=root/'broken';p.mkdir();(p/'result.json').write_bytes(b'\xff')
            report,_=collect(root,'b')
            self.assertEqual(report['invalid_record_details'][0]['error_type'],'UnicodeDecodeError')

if __name__=='__main__':unittest.main()
