import json
from pathlib import Path

def load():
    return json.loads((Path(__file__).with_name("context.json")).read_text())

def run():
    import tempfile
    from trialvault.service import Vault
    from trialvault.validation import canonical
    data = load()
    with tempfile.TemporaryDirectory() as directory:
        with Vault(directory) as app:
            fields = [{'name':n,'type':t,'nullable':n=='right_day'} for n,t in [('id','text'),('batch','text'),('temperature_c','decimal'),('entry_day','decimal'),('left_day','decimal'),('right_day','decimal'),('kind','text')]]
            app.registry.add_schema({'id':'aging','version':1,'fields':fields,'primary_key':['id']})
            app.registry.register('seal-aging','合成密封材料老化','aging','engineer')
            content = ''.join(canonical(r)+'\n' for r in data['records']).encode()
            result = app.ingestion.ingest('aging-1','seal-aging',1,content,'jsonl','internal',['research'])
            app.releases.plan('aging-approved',[{'dataset':'seal-aging','version':1}],'research','internal')
            app.releases.approve('aging-approved',1,'independent-reviewer','巡检记录核对')
            return {'ok': True, 'records': result['rows']}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
