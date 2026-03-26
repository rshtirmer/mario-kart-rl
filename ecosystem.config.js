module.exports = {
  apps: [{
    name: 'mario-kart-dashboard',
    script: 'src/mariokart/dashboard.py',
    interpreter: '/Users/dev/projects/mario-kart-rl/.venv/bin/python',
    cwd: '/Users/dev/projects/mario-kart-rl',
    env: {
      PYTHONUNBUFFERED: '1'
    },
    watch: ['src/mariokart/dashboard.py'],
    restart_delay: 2000,
    max_restarts: 10,
  }]
};
