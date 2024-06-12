FROM python:3.11

LABEL maintainer="Sebastian Kempf"
LABEL version="0.1"
LABEL description="Dockerfile for GlassPatentExtractor"

WORKDIR /usr/src/server

COPY requirements.txt .
# install requirements
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
# RUN pip3 install --no-cache-dir git+https://gitlab+deploy-token-113:pdXYgWy8yHMq5it7CxK6@gitlab2.informatik.uni-wuerzburg.de/nof36hn/guppyocr.git@tensorboard
# RUN pip3 install --no-cache-dir albumentations apscheduler Flask-ApScheduler


# install local modules
COPY kieta-data-objs kieta-data-objs
RUN pip3 install --no-cache-dir kieta-data-objs/.
# RUN pip3 install --no-cache-dir git+https://gitlab+deploy-token-138:gldt-SntLC9xSsC3vzFNPGLUH@gitlab2.informatik.uni-wuerzburg.de/sek50xe/kieta_data_objs.git
COPY kieta-modules kieta-modules
RUN pip3 install --no-cache-dir kieta-modules/.


# copy project
COPY server-glass /usr/src/server

# flask
EXPOSE 5000

ENTRYPOINT [ "flask", "run", "--host=0.0.0.0", "--port=5000" ]