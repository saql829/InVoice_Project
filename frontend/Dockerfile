# frontend/Dockerfile
FROM node:18

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# build static files
RUN npm run build

EXPOSE 5173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "5173"]
