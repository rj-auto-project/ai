import os
import shutil
import paramiko
from scp import SCPClient
from dotenv import load_dotenv

# Step 1: Zip the folder
def zip_folder(folder_path, output_zip):
    shutil.make_archive(output_zip, 'zip', folder_path)
    print(f"Folder {folder_path} zipped as {output_zip}.zip")

# Step 2: Upload the zip file to VM using SCP with SSH key
def upload_to_vm_with_key(hostname, username, ssh_key_path, local_file, remote_path):
    # SSH and SCP setup
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print("Connecting to VM using SSH key...")
        # Connect to the VM using SSH private key
        ssh.connect(hostname, username=username, key_filename=ssh_key_path)
        
        with SCPClient(ssh.get_transport()) as scp:
            print(f"Uploading {local_file} to {remote_path}...")
            scp.put(local_file, remote_path)
            print("Upload completed!")
    except Exception as e:
        print(f"Error during upload: {e}")
    finally:
        ssh.close()

# Main function to zip and upload
def zip_and_upload(folder_path, hostname, username, ssh_key_path, remote_path):
    output_zip = folder_path.rstrip(os.sep)  # Remove trailing slash for clean zip name
    zip_folder(folder_path, output_zip)
    
    # Upload to VM
    local_zip_file = f"{output_zip}.zip"
    upload_to_vm_with_key(hostname, username, ssh_key_path, local_zip_file, remote_path)

if __name__ == "__main__":
    load_dotenv()
    # Example usage:
    folder_path = os.getenv("FOLDER_PATH")
    vm_hostname = os.getenv("VM_HOSTNAME")
    vm_username = os.getenv("VM_USER")
    ssh_key_path = os.getenv("SSH_KEY") # e.g., "~/.ssh/id_rsa"
    remote_path = os.getenv("REMOTE_PATH")

    # Zip and upload the folder
    zip_and_upload(folder_path, vm_hostname, vm_username, ssh_key_path, remote_path)