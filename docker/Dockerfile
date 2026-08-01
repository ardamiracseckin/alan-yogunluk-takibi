# Konteynerde GUI yok: opencv-python yerine headless sürüm kurulur.
# CPU'da çalışacağı için torch'un CPU tekerleği kullanılır (imaj ~1 GB yerine
# ~4 GB olmasın diye).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OCCUPANCY_HOST=0.0.0.0

WORKDIR /app

# opencv-headless bile libGL olmadan bazı kodekleri açamıyor; ffmpeg
# kütüphaneleri video dosyası ve RTSP okumak için gerekli.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless

COPY occupancy/ occupancy/
COPY web/ web/
COPY tools/ tools/
COPY ornek/ ornek/

EXPOSE 8000

CMD ["python", "-m", "occupancy"]
