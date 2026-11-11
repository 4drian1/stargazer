import os

def has(f, keys):
    try:
        t = open(f, encoding='utf-8').read()
    except Exception as e:
        return 'ERR ' + str(e)
    return {k: (k in t) for k in keys}

print('server.py', has('server.py', ['api_calendar_next', 'api_plan_versions',
      'api_global_overview', '_enrich_execution_status', 'import ui_calendar_api']))
print('assistant_workflow.py', has('assistant_workflow.py',
      ['confirmed_status', "confirmed'", 'def state']))
print('app.py', has('app.py', ['--acceptance', 'acceptance_ui_integration_v1']))
print('ui_backend_config.py', has('ui_backend_config.py',
      ['paths.APP_DIR', 'import paths']))
ih = 'web/index.html'
if os.path.exists(ih):
    t = open(ih, encoding='utf-8').read()
    print('web/index.html refs new bundle?', 'index-CIOVEJrB' in t)
print('web assets:', sorted(os.listdir('web/assets')) if os.path.isdir('web/assets') else 'NA')
