"""
AWS S3 DOCUMENT STORAGE FOR LAW FIRM CHATBOT
Secure document upload, download, and management
"""

import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import mimetypes
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")

# Folder structure in S3
FOLDERS = {
    "intake_forms": "intake-forms",
    "case_documents": "case-documents",
    "client_uploads": "client-uploads",
    "signed_agreements": "signed-agreements",
    "medical_records": "medical-records",
    "photos": "photos",
    "other": "other-documents"
}

# ============================================
# S3 CLIENT
# ============================================

def get_s3_client():
    """Get configured S3 client"""
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
        raise ValueError("Missing AWS credentials or bucket name in environment variables")
    
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


def check_bucket_exists() -> bool:
    """Check if S3 bucket exists and is accessible"""
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        return True
    except ClientError:
        return False


# ============================================
# UPLOAD FUNCTIONS
# ============================================

def upload_file_to_s3(
    file_path: str,
    client_id: str,
    case_id: Optional[str] = None,
    folder_type: str = "client_uploads",
    custom_filename: Optional[str] = None
) -> Dict[str, str]:
    """
    Upload a file to S3
    
    Args:
        file_path: Local path to file
        client_id: Client identifier
        case_id: Optional case identifier
        folder_type: Type of folder (from FOLDERS dict)
        custom_filename: Optional custom filename
    
    Returns:
        dict: Contains 's3_key', 'url', 'bucket'
    """
    try:
        s3 = get_s3_client()
        
        # Generate S3 key (path in bucket)
        filename = custom_filename or Path(file_path).name
        
        if case_id:
            s3_key = f"{FOLDERS[folder_type]}/{client_id}/{case_id}/{filename}"
        else:
            s3_key = f"{FOLDERS[folder_type]}/{client_id}/{filename}"
        
        # Get file metadata
        content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        file_size = os.path.getsize(file_path)
        
        print(f"📤 Uploading {filename} to S3...")
        print(f"   Size: {file_size / 1024:.2f} KB")
        print(f"   Key: {s3_key}")
        
        # Upload file
        s3.upload_file(
            file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                'ContentType': content_type,
                'Metadata': {
                    'client_id': client_id,
                    'case_id': case_id or '',
                    'uploaded_at': datetime.utcnow().isoformat()
                }
            }
        )
        
        print(f"✅ Upload successful!")
        
        return {
            's3_key': s3_key,
            'bucket': S3_BUCKET_NAME,
            'url': f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}",
            'filename': filename,
            'size': file_size
        }
        
    except ClientError as e:
        print(f"❌ S3 upload failed: {str(e)}")
        raise Exception(f"Failed to upload file to S3: {str(e)}")


def upload_file_object_to_s3(
    file_content: bytes,
    filename: str,
    client_id: str,
    case_id: Optional[str] = None,
    folder_type: str = "client_uploads",
    content_type: str = "application/octet-stream"
) -> Dict[str, str]:
    """
    Upload file content directly to S3 (from API upload)
    
    Args:
        file_content: File bytes
        filename: Name of file
        client_id: Client identifier
        case_id: Optional case identifier
        folder_type: Type of folder
        content_type: MIME type
    
    Returns:
        dict: Contains 's3_key', 'url', 'bucket'
    """
    try:
        s3 = get_s3_client()
        
        # Generate S3 key
        if case_id:
            s3_key = f"{FOLDERS[folder_type]}/{client_id}/{case_id}/{filename}"
        else:
            s3_key = f"{FOLDERS[folder_type]}/{client_id}/{filename}"
        
        print(f"📤 Uploading {filename} to S3...")
        print(f"   Size: {len(file_content) / 1024:.2f} KB")
        print(f"   Key: {s3_key}")
        
        # Upload file content
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type,
            Metadata={
                'client_id': client_id,
                'case_id': case_id or '',
                'uploaded_at': datetime.utcnow().isoformat()
            }
        )
        
        print(f"✅ Upload successful!")
        
        return {
            's3_key': s3_key,
            'bucket': S3_BUCKET_NAME,
            'url': f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}",
            'filename': filename,
            'size': len(file_content)
        }
        
    except ClientError as e:
        print(f"❌ S3 upload failed: {str(e)}")
        raise Exception(f"Failed to upload file to S3: {str(e)}")


# ============================================
# PRESIGNED URL GENERATION
# ============================================

def generate_presigned_upload_url(
    filename: str,
    client_id: str,
    case_id: Optional[str] = None,
    folder_type: str = "client_uploads",
    expiration: int = 3600
) -> Dict[str, str]:
    """
    Generate presigned URL for direct browser upload to S3
    
    Args:
        filename: Name of file to upload
        client_id: Client identifier
        case_id: Optional case identifier
        folder_type: Type of folder
        expiration: URL expiration in seconds (default 1 hour)
    
    Returns:
        dict: Contains 'upload_url', 's3_key', 'fields' for POST
    """
    try:
        s3 = get_s3_client()
        
        # Generate S3 key
        if case_id:
            s3_key = f"{FOLDERS[folder_type]}/{client_id}/{case_id}/{filename}"
        else:
            s3_key = f"{FOLDERS[folder_type]}/{client_id}/{filename}"
        
        # Generate presigned POST
        response = s3.generate_presigned_post(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Fields={
                'x-amz-meta-client_id': client_id,
                'x-amz-meta-case_id': case_id or '',
                'x-amz-meta-uploaded_at': datetime.utcnow().isoformat()
            },
            Conditions=[
                ['content-length-range', 1, 104857600]  # 1 byte to 100 MB
            ],
            ExpiresIn=expiration
        )
        
        print(f"✅ Generated presigned upload URL for {filename}")
        print(f"   Expires in {expiration} seconds")
        
        return {
            'upload_url': response['url'],
            's3_key': s3_key,
            'fields': response['fields']
        }
        
    except ClientError as e:
        print(f"❌ Failed to generate presigned URL: {str(e)}")
        raise Exception(f"Failed to generate presigned URL: {str(e)}")


def generate_presigned_download_url(
    s3_key: str,
    expiration: int = 3600
) -> str:
    """
    Generate presigned URL for downloading a file
    
    Args:
        s3_key: S3 object key
        expiration: URL expiration in seconds (default 1 hour)
    
    Returns:
        str: Presigned download URL
    """
    try:
        s3 = get_s3_client()
        
        url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET_NAME,
                'Key': s3_key
            },
            ExpiresIn=expiration
        )
        
        print(f"✅ Generated presigned download URL")
        print(f"   Expires in {expiration} seconds")
        
        return url
        
    except ClientError as e:
        print(f"❌ Failed to generate download URL: {str(e)}")
        raise Exception(f"Failed to generate download URL: {str(e)}")


# ============================================
# FILE MANAGEMENT
# ============================================

def list_client_files(
    client_id: str,
    case_id: Optional[str] = None,
    folder_type: Optional[str] = None
) -> List[Dict[str, any]]:
    """
    List all files for a client or case
    
    Args:
        client_id: Client identifier
        case_id: Optional case identifier
        folder_type: Optional folder type filter
    
    Returns:
        list: List of file metadata dicts
    """
    try:
        s3 = get_s3_client()
        
        # Build prefix
        if folder_type:
            if case_id:
                prefix = f"{FOLDERS[folder_type]}/{client_id}/{case_id}/"
            else:
                prefix = f"{FOLDERS[folder_type]}/{client_id}/"
        else:
            prefix = f"{client_id}/" if not case_id else f"{client_id}/{case_id}/"
        
        print(f"📂 Listing files with prefix: {prefix}")
        
        # List objects
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=prefix
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                # Get metadata
                metadata_response = s3.head_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=obj['Key']
                )
                
                files.append({
                    's3_key': obj['Key'],
                    'filename': Path(obj['Key']).name,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'content_type': metadata_response.get('ContentType', 'unknown'),
                    'metadata': metadata_response.get('Metadata', {}),
                    'download_url': generate_presigned_download_url(obj['Key'], expiration=3600)
                })
        
        print(f"✅ Found {len(files)} files")
        return files
        
    except ClientError as e:
        print(f"❌ Failed to list files: {str(e)}")
        raise Exception(f"Failed to list files: {str(e)}")


def delete_file(s3_key: str) -> bool:
    """
    Delete a file from S3
    
    Args:
        s3_key: S3 object key
    
    Returns:
        bool: True if successful
    """
    try:
        s3 = get_s3_client()
        
        print(f"🗑️ Deleting file: {s3_key}")
        
        s3.delete_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )
        
        print(f"✅ File deleted successfully")
        return True
        
    except ClientError as e:
        print(f"❌ Failed to delete file: {str(e)}")
        raise Exception(f"Failed to delete file: {str(e)}")


def download_file(s3_key: str, local_path: str) -> str:
    """
    Download a file from S3 to local path
    
    Args:
        s3_key: S3 object key
        local_path: Local path to save file
    
    Returns:
        str: Local file path
    """
    try:
        s3 = get_s3_client()
        
        print(f"📥 Downloading file: {s3_key}")
        
        s3.download_file(
            S3_BUCKET_NAME,
            s3_key,
            local_path
        )
        
        print(f"✅ Downloaded to: {local_path}")
        return local_path
        
    except ClientError as e:
        print(f"❌ Failed to download file: {str(e)}")
        raise Exception(f"Failed to download file: {str(e)}")


def get_file_metadata(s3_key: str) -> Dict[str, any]:
    """
    Get metadata for a file
    
    Args:
        s3_key: S3 object key
    
    Returns:
        dict: File metadata
    """
    try:
        s3 = get_s3_client()
        
        response = s3.head_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )
        
        return {
            's3_key': s3_key,
            'filename': Path(s3_key).name,
            'size': response['ContentLength'],
            'last_modified': response['LastModified'].isoformat(),
            'content_type': response.get('ContentType', 'unknown'),
            'metadata': response.get('Metadata', {})
        }
        
    except ClientError as e:
        print(f"❌ Failed to get file metadata: {str(e)}")
        raise Exception(f"Failed to get file metadata: {str(e)}")


# ============================================
# BUCKET SETUP
# ============================================

def create_bucket_if_not_exists():
    """
    Create S3 bucket if it doesn't exist
    Only call this during initial setup
    """
    try:
        s3 = get_s3_client()
        
        if check_bucket_exists():
            print(f"✅ Bucket {S3_BUCKET_NAME} already exists")
            return True
        
        print(f"📦 Creating bucket: {S3_BUCKET_NAME}")
        
        if AWS_REGION == 'us-east-1':
            s3.create_bucket(Bucket=S3_BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET_NAME,
                CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
            )
        
        # Set bucket encryption
        s3.put_bucket_encryption(
            Bucket=S3_BUCKET_NAME,
            ServerSideEncryptionConfiguration={
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        }
                    }
                ]
            }
        )
        
        # Block public access
        s3.put_public_access_block(
            Bucket=S3_BUCKET_NAME,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': True,
                'RestrictPublicBuckets': True
            }
        )
        
        print(f"✅ Bucket created with encryption and security enabled")
        return True
        
    except ClientError as e:
        print(f"❌ Failed to create bucket: {str(e)}")
        raise Exception(f"Failed to create bucket: {str(e)}")


# ============================================
# TESTING
# ============================================

def test_s3_connection() -> Dict[str, any]:
    """Test S3 connection and return status"""
    try:
        s3 = get_s3_client()
        
        # Check bucket exists
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        
        # List objects (just to verify access)
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            MaxKeys=1
        )
        
        total_objects = response.get('KeyCount', 0)
        
        return {
            'success': True,
            'message': 'Successfully connected to AWS S3!',
            'bucket': S3_BUCKET_NAME,
            'region': AWS_REGION,
            'total_files': total_objects
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'bucket': S3_BUCKET_NAME,
            'region': AWS_REGION
        }


if __name__ == "__main__":
    """
    Test S3 connection
    Run: python s3_storage.py
    """
    
    print("\n" + "=" * 60)
    print("AWS S3 STORAGE TEST")
    print("=" * 60)
    
    result = test_s3_connection()
    
    if result['success']:
        print(f"\n✅ {result['message']}")
        print(f"   Bucket: {result['bucket']}")
        print(f"   Region: {result['region']}")
        print(f"   Files: {result['total_files']}")
    else:
        print(f"\n❌ Connection failed: {result['error']}")
        print("\nTroubleshooting:")
        print("  1. Set AWS_ACCESS_KEY_ID in environment")
        print("  2. Set AWS_SECRET_ACCESS_KEY in environment")
        print("  3. Set S3_BUCKET_NAME in environment")
        print("  4. Verify IAM user has S3 permissions")
    
    print("\n" + "=" * 60)