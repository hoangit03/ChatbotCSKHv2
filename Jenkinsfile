// ══════════════════════════════════════════════════════════════
// Jenkinsfile — ChatbotCSKHv2 (Tầng 3)
// Trigger: GitHub push → git pull → docker compose build → deploy
// ══════════════════════════════════════════════════════════════

pipeline {
    agent any
    
    environment {
        PROJECT_DIR = '/root/ChatbotCSKHv2'
        COMPOSE_FILE = 'docker-compose.yml'
        APP_CONTAINER = 'cskh_api'
        HEALTH_URL = 'http://localhost:8006/health'
    }

    stages {
        stage('Pull Latest Code') {
            steps {
                sh """
                    cd ${PROJECT_DIR}
                    git fetch origin
                    git reset --hard origin/dev
                """
            }
        }

        stage('Build & Deploy') {
            steps {
                sh """
                    cd ${PROJECT_DIR}
                    docker-compose up -d --build --force-recreate
                """
            }
        }

        stage('Wait for Healthy') {
            steps {
                sh """
                    echo "Waiting for ${APP_CONTAINER} to be healthy..."
                    for i in \$(seq 1 20); do
                        STATUS=\$(docker inspect --format='{{.State.Health.Status}}' ${APP_CONTAINER} 2>/dev/null || echo "starting")
                        echo "  Attempt \$i/20: \$STATUS"
                        if [ "\$STATUS" = "healthy" ]; then
                            echo "✅ Container is healthy!"
                            exit 0
                        fi
                        sleep 10
                    done
                    echo "❌ Health check timeout!"
                    docker logs --tail 20 ${APP_CONTAINER}
                    exit 1
                """
            }
        }

        stage('Reload Nginx') {
            steps {
                sh 'docker exec global_nginx nginx -s reload'
            }
        }

        stage('Smoke Test') {
            steps {
                sh """
                    RESPONSE=\$(curl -sf ${HEALTH_URL} || echo "FAILED")
                    echo "Health response: \$RESPONSE"
                    echo \$RESPONSE | grep -q '"healthy"' || (echo "❌ Smoke test failed!" && exit 1)
                    echo "✅ Smoke test passed!"
                """
            }
        }
    }

    post {
        success {
            echo '🎉 ChatbotCSKHv2 deployed successfully!'
        }
        failure {
            echo '❌ Deployment failed! Check logs above.'
        }
    }
}
