"""
Secrets Loader for Project HOLLYWOOD
Loads API keys and secrets from SECRETS file
"""

import os
from pathlib import Path
from typing import Dict, Optional

def load_secrets(secrets_file: str = 'SECRETS') -> Dict[str, str]:
    """
    Load secrets from SECRETS file.

    Args:
        secrets_file: Path to secrets file (default: 'SECRETS')

    Returns:
        Dictionary of key-value pairs
    """
    secrets = {}

    # Try to find SECRETS file
    script_dir = Path(__file__).parent
    secrets_path = script_dir / secrets_file

    if not secrets_path.exists():
        # Try project root (one level above Dashboard/)
        secrets_path = script_dir.parent / secrets_file

    if not secrets_path.exists():
        # Try current directory
        secrets_path = Path(secrets_file)

    if not secrets_path.exists():
        print(f"⚠️  SECRETS file not found at {secrets_path}")
        print(f"   Copy SECRETS.template to SECRETS and add your API keys")
        return secrets

    # Parse secrets file
    with open(secrets_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Skip placeholder values
                if value and not value.startswith('your_') and value != 'your_key_here':
                    secrets[key] = value

    return secrets

def get_secret(key: str, default: Optional[str] = None, secrets_file: str = 'SECRETS') -> Optional[str]:
    """
    Get a single secret by key.

    Priority:
    1. Environment variable (if set)
    2. SECRETS file
    3. Default value

    Args:
        key: Secret key name
        default: Default value if not found
        secrets_file: Path to secrets file

    Returns:
        Secret value or default
    """
    # Try environment variable first
    env_value = os.getenv(key)
    if env_value:
        return env_value

    # Try SECRETS file
    secrets = load_secrets(secrets_file)
    return secrets.get(key, default)

def check_required_secrets(required_keys: list, secrets_file: str = 'SECRETS') -> bool:
    """
    Check if all required secrets are present.

    Args:
        required_keys: List of required secret keys
        secrets_file: Path to secrets file

    Returns:
        True if all required secrets present, False otherwise
    """
    secrets = load_secrets(secrets_file)
    missing = []

    for key in required_keys:
        # Check env var first
        if os.getenv(key):
            continue

        # Check SECRETS file
        if key not in secrets:
            missing.append(key)

    if missing:
        print("❌ Missing required secrets:")
        for key in missing:
            print(f"   - {key}")
        print(f"\nAdd these to your SECRETS file or set as environment variables")
        return False

    return True

# Convenience function to setup secrets for interactive use
def setup_secrets_env(secrets_file: str = 'SECRETS'):
    """
    Load secrets and set as environment variables.
    Useful for Jupyter notebooks.

    Usage:
        from secrets_loader import setup_secrets_env
        setup_secrets_env()
    """
    secrets = load_secrets(secrets_file)

    if not secrets:
        print("⚠️  No secrets loaded")
        return

    count = 0
    for key, value in secrets.items():
        if not os.getenv(key):  # Don't override existing env vars
            os.environ[key] = value
            count += 1

    print(f"✓ Loaded {count} secrets into environment")

if __name__ == '__main__':
    # Test the loader
    print("Testing secrets loader...")
    secrets = load_secrets()

    if secrets:
        print(f"\n✓ Found {len(secrets)} secrets:")
        for key in secrets.keys():
            # Don't print actual values!
            print(f"  - {key}: {'*' * 8}")
    else:
        print("\n⚠️  No secrets found. Copy SECRETS.template to SECRETS and add your keys.")
