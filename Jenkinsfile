pipeline {
    agent any
    environment {
        ECR_REPO    = "${env.ENV}-${env.TEAM}-ner-microservice"
        IMAGE_TAG   = "${env.BUILD_NUMBER}"
    }
    stages {
        stage('Configuration') {
            steps {
				echo "Updating Configs"
                sh "sed -i 's@<ENV>@$ENV@g' Dockerfile"
                sh "sed -i 's@<ENV>@$ENV@g' app/constants.py"
                sh "sed -i 's@<JSL_SPARK_NLP_LICENSE>@$JSL_SPARK_NLP_LICENSE@g' Dockerfile"
                sh "sed -i 's@<JSL_SPARK_LICENSE_SECRET_VERSION_3_1_1>@$JSL_SPARK_LICENSE_SECRET_VERSION_3_1_1@g' Dockerfile"
                sh "sed -i 's@<MODELS_AWS_ACCESS_KEY>@$MODELS_AWS_ACCESS_KEY@g' app/constants.py"
                sh "sed -i 's@<MODELS_AWS_SECRET_KEY>@$MODELS_AWS_SECRET_KEY@g' app/constants.py"
                sh "sed -i 's@<MODELS_AWS_REGION>@$MODELS_AWS_REGION@g' app/constants.py"
                sh "sed -i 's@<MODELS_BUCKET_NAME>@$MODELS_BUCKET_NAME@g' app/constants.py"
                sh "sed -i 's@<VR_SPARK_3_MODEL_FOLDER_NAME>@$VR_SPARK_3_MODEL_FOLDER_NAME@g' app/constants.py"
                // sh "sed -i 's@<NER_FILES_BUCKET_NAME>@$NER_FILES_BUCKET_NAME@g' app/constants.py"
                // sh "sed -i 's@<NER_OUTPUT_FOLDER_NAME>@$NER_OUTPUT_FOLDER_NAME@g' app/constants.py"
            }
        }
        stage('Build') {
            steps {
                sh 'ECR=$(aws ecr get-login --no-include-email --region $AWS_REGION) && $ECR'
                sh 'docker build -t $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG .'
                sh 'docker push $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG'
            }
        }
        stage('Deploy') {
            steps {
            	echo "Deploying"
                sh "sed -i 's@<ACCOUNT>@$AWS_ACCOUNT@g' kubernetes_scripts/deployment.yaml"
                sh "sed -i 's@<REGION>@$AWS_REGION@g' kubernetes_scripts/deployment.yaml"
                sh "sed -i 's@<REPO>@$ECR_REPO@g' kubernetes_scripts/deployment.yaml"
                sh "sed -i 's@<TAG>@$IMAGE_TAG@g' kubernetes_scripts/deployment.yaml"
                sh 'kubectl delete -f kubernetes_scripts/deployment.yaml -n $ENV || true'
                sleep time: 30, unit: 'SECONDS'
                sh 'kubectl apply -f kubernetes_scripts/ -n $ENV'          }
        }
    }
}
