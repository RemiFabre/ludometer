FROM node:22-slim
WORKDIR /app
COPY package.json server.js verify.js engine.js ./
ENV PORT=7860
EXPOSE 7860
CMD ["node", "server.js"]
