#!/bin/bash
echo "Stopping all containers..."
cd /root/chatbot && docker-compose -f docker-compose.server.yml down
cd /root/ChatbotCSKHv2 && docker compose down

echo "Pruning system to free up space..."
docker system prune -a -f
docker builder prune -a -f

echo "Starting ChatbotCSKHv2..."
cd /root/ChatbotCSKHv2 && docker compose up -d --build

echo "Starting Chatbot..."
cd /root/chatbot && docker-compose -f docker-compose.server.yml up -d --build

echo "Done!"
