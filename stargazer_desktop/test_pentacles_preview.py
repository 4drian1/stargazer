"""Local visual QA server only; no production server, notifications or monitor imports."""
from pathlib import Path
from flask import Flask, jsonify, send_from_directory
import pentacles_demo

WEB = Path(__file__).parent / 'web'
app = Flask('pentacles-preview')


@app.route('/')
def index():
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    return html.replace('</body>', "<script>window.addEventListener('load',()=>{switchPage('pentacles');refreshPentacles();if(new URLSearchParams(location.search).has('journal'))document.getElementById('pentaclesJournal').click();});</script></body>")


@app.route('/api/pentacles/state')
def state():
    return jsonify(pentacles_demo.state())


@app.route('/api/<path:path>', methods=['GET', 'POST'])
def placeholder(path):
    return jsonify({})


@app.route('/api/pentacles/journal')
def journal():
    return jsonify(pentacles_demo.monitor.journal.report())


@app.route('/<path:path>')
def assets(path):
    return send_from_directory(WEB, path)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=17939, use_reloader=False)
