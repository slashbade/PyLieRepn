conda activate lie_repn_env
pkill -9 uwsgi
git restore .
git pull
uwsgi --ini uwsgi.ini
echo "Update complete"