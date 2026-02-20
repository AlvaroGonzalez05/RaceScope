# 1. Usamos Node 20 (Necesario para Vite actual)
FROM nikolaik/python-nodejs:python3.10-nodejs20

WORKDIR /app

# 2. Instalamos dependencias de Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. Instalamos dependencias de Node (Backend)
COPY backend/package.json backend/
WORKDIR /app/backend
RUN npm install

# 4. Instalamos dependencias de Node (Frontend)
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install

# -----------------------------------------------------------
# 5. Copiamos el código fuente
# -----------------------------------------------------------
WORKDIR /app
COPY . .

# -----------------------------------------------------------
# 6. FIX: Forzamos instalación de librerías faltantes y construimos
# -----------------------------------------------------------
WORKDIR /app/frontend
# ¡ESTA ES LA LÍNEA NUEVA QUE ARREGLA EL ERROR! 👇
RUN npm install axios recharts
RUN npm run build

# 7. Movemos la carpeta 'dist' generada al backend
WORKDIR /app
RUN rm -rf backend/public && mv frontend/dist backend/public

# 8. Arrancamos
EXPOSE 5001
WORKDIR /app/backend
CMD ["node", "server.js"]