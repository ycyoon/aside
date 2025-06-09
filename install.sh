pip install --upgrade pip
pip install -r requirements.txt
# This needs to be installed separately, after torch and psutil
pip install flash-attn==2.7.4.post1 --no-build-isolation
# Build deepspeed to allow optimizer cpu offloading
DS_BUILD_CPU_ADAM=1 pip install deepspeed==0.17.0 --no-build-isolation