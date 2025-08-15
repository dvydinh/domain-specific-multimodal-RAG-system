# Stage 1: Build React/Vite App using Node
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies first for cache layer
COPY frontend/package.json ./
# Use package-lock.json if it exists, otherwise just npm install
COPY frontend/package-lock.json* ./
RUN npm install

# Copy frontend source code and compile optimized static assets
COPY frontend/ ./
RUN npm run build

# Stage 2: Serve statically with lightweight NGINX Alpine
FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy static assets from builder stage
COPY --from=build /app/dist /usr/share/nginx/html

# Replace with our custom nginx.conf
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 for traffic
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
