# The chat UI, built once and served as static files by nginx.
#
# Vite inlines VITE_* at BUILD time — there is no runtime config to change later —
# so VITE_API_BASE_URL is deliberately left EMPTY here. api.ts then falls back to
# `location.origin`, and nginx proxies /models and /chat to the backend container
# (see nginx.conf). One origin for the page and the API means no CORS, and it works
# unchanged whether you open it on localhost or from another machine on the LAN.

# ------------------------------------------------------------------ build ---
FROM node:20-alpine AS build

WORKDIR /app

# package-lock.json is committed, so `npm ci` gets the exact tree the app was
# developed against. Copied on its own: the install layer then survives every
# source edit.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# The product name in the sidebar and empty state. The API base is NOT an arg —
# see the note at the top.
ARG VITE_APP_NAME=HFrag
ENV VITE_APP_NAME=$VITE_APP_NAME

# `npm run build` is `tsc -b && vite build`, so a type error fails the image
# rather than shipping a stale bundle.
RUN npm run build

# ------------------------------------------------------------------ serve ---
FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
