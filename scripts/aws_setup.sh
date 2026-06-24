#!/bin/bash
set -euo pipefail

AWS_REGION="eu-west-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="fraud-detection"
CLUSTER_NAME="fraud-detection-cluster"
NAMESPACE="fraud-detection"
NODE_TYPE="t3.medium"
MIN_NODES=2
MAX_NODES=5

echo "Account: $AWS_ACCOUNT_ID"
echo "Region:  $AWS_REGION"

# 1. Create ECR repository
echo "[1/5] Creating ECR repository..."
aws ecr create-repository \
    --repository-name $ECR_REPO \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || echo "ECR repo already exists"

# 2. Create EKS cluster
echo "[2/5] Creating EKS cluster (~15 minutes)..."
eksctl create cluster \
    --name $CLUSTER_NAME \
    --region $AWS_REGION \
    --nodegroup-name standard-workers \
    --node-type $NODE_TYPE \
    --nodes-min $MIN_NODES \
    --nodes-max $MAX_NODES \
    --managed \
    --asg-access \
    --alb-ingress-access \
    2>/dev/null || echo "Cluster already exists"

# 3. Update kubeconfig
echo "[3/5] Configuring kubectl..."
aws eks update-kubeconfig \
    --name $CLUSTER_NAME \
    --region $AWS_REGION

# 4. Install AWS Load Balancer Controller
echo "[4/5] Installing Load Balancer Controller..."
aws iam create-policy \
    --policy-name AWSLoadBalancerControllerIAMPolicy \
    --policy-document https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.7.1/docs/install/iam_policy.json \
    2>/dev/null || echo "IAM policy already exists"

eksctl create iamserviceaccount \
    --cluster=$CLUSTER_NAME \
    --namespace=kube-system \
    --name=aws-load-balancer-controller \
    --attach-policy-arn=arn:aws:iam::$AWS_ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy \
    --override-existing-serviceaccounts \
    --approve \
    2>/dev/null || echo "Service account already exists"

helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n kube-system \
    --set clusterName=$CLUSTER_NAME \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller

# 5. Create namespace
echo "[5/5] Creating namespace..."
kubectl create namespace $NAMESPACE 2>/dev/null || echo "Namespace already exists"

echo ""
echo "Setup complete!"
echo "ECR URI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"