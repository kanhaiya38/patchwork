#!/usr/bin/env python3
"""
Test script to verify experience replay buffer data integrity.
Checks that all required fields are preserved during save/load operations.
"""

import sys
import logging
from pathlib import Path
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_replay_buffer_fields():
    """Test that replay buffer preserves all necessary fields."""

    # Import the ExperienceReplayBuffer from training.py
    sys.path.insert(0, str(Path(__file__).parent))
    from training import ExperienceReplayBuffer

    logger.info("=" * 80)
    logger.info("TEST: Experience Replay Buffer Field Preservation")
    logger.info("=" * 80)

    # Create a mock tokenized dataset with all required fields
    mock_data = {
        "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        "attention_mask": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "answer": ["Answer 1", "Answer 2", "Answer 3"],
    }
    mock_dataset = Dataset.from_dict(mock_data)

    logger.info(f"\n1. Created mock dataset with {len(mock_dataset)} samples")
    logger.info(f"   Fields: {mock_dataset.column_names}")
    logger.info(f"   Sample 0: {mock_dataset[0]}")

    # Create replay buffer and add samples
    buffer = ExperienceReplayBuffer(max_samples_per_task=2, selection_strategy="first")
    buffer.add_task_samples(task_id=0, dataset=mock_dataset, task_name="TestTask")

    logger.info(f"\n2. Added samples to replay buffer")
    logger.info(f"   Buffer stats: {buffer.get_stats()}")

    # Check that samples in buffer have all fields
    stored_samples = buffer.buffer[0]
    logger.info(f"\n3. Checking stored samples in memory")
    logger.info(f"   Stored fields: {stored_samples.column_names}")
    logger.info(f"   Sample 0: {stored_samples[0]}")

    # Verify all required fields are present
    required_fields = {"input_ids", "attention_mask", "answer"}
    stored_fields = set(stored_samples.column_names)

    if required_fields.issubset(stored_fields):
        logger.info(f"   ✓ All required fields present: {required_fields}")
    else:
        missing = required_fields - stored_fields
        logger.error(f"   ✗ Missing fields: {missing}")
        return False

    # Test save/load cycle
    temp_path = Path("./test_replay_buffer_temp")
    try:
        logger.info(f"\n4. Testing save/load cycle")
        buffer.save(temp_path)
        logger.info(f"   ✓ Buffer saved to {temp_path}")

        # Load buffer
        loaded_buffer = ExperienceReplayBuffer.load(temp_path)
        if loaded_buffer is None:
            logger.error(f"   ✗ Failed to load buffer")
            return False

        logger.info(f"   ✓ Buffer loaded from {temp_path}")
        logger.info(f"   Loaded stats: {loaded_buffer.get_stats()}")

        # Check loaded samples
        loaded_samples = loaded_buffer.buffer[0]
        logger.info(f"\n5. Checking loaded samples")
        logger.info(f"   Loaded fields: {loaded_samples.column_names}")
        logger.info(f"   Sample 0: {loaded_samples[0]}")

        loaded_fields = set(loaded_samples.column_names)
        if required_fields.issubset(loaded_fields):
            logger.info(f"   ✓ All required fields present after load: {required_fields}")
        else:
            missing = required_fields - loaded_fields
            logger.error(f"   ✗ Missing fields after load: {missing}")
            return False

        # Test get_replay_dataset (simulating task 1 getting replay from task 0)
        logger.info(f"\n6. Testing get_replay_dataset()")
        replay_dataset = loaded_buffer.get_replay_dataset(current_task_id=1)

        if replay_dataset is None:
            logger.error(f"   ✗ get_replay_dataset returned None")
            return False

        logger.info(f"   Replay dataset size: {len(replay_dataset)}")
        logger.info(f"   Replay fields: {replay_dataset.column_names}")
        logger.info(f"   Sample 0: {replay_dataset[0]}")

        replay_fields = set(replay_dataset.column_names)
        if required_fields.issubset(replay_fields):
            logger.info(f"   ✓ All required fields present in replay dataset: {required_fields}")
        else:
            missing = required_fields - replay_fields
            logger.error(f"   ✗ Missing fields in replay dataset: {missing}")
            return False

        # Verify data integrity
        logger.info(f"\n7. Verifying data integrity")
        original_answer = stored_samples[0]["answer"]
        loaded_answer = loaded_samples[0]["answer"]
        replay_answer = replay_dataset[0]["answer"]

        logger.info(f"   Original answer: {original_answer}")
        logger.info(f"   Loaded answer: {loaded_answer}")
        logger.info(f"   Replay answer: {replay_answer}")

        if original_answer == loaded_answer == replay_answer:
            logger.info(f"   ✓ Data integrity verified - answers match")
        else:
            logger.error(f"   ✗ Data integrity failed - answers don't match")
            return False

    finally:
        # Cleanup
        if temp_path.exists():
            import shutil
            shutil.rmtree(temp_path)
            logger.info(f"\n8. Cleaned up temporary files")

    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    return True


def test_with_actual_checkpoint():
    """Test with actual checkpoint data if available."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Checking Actual Replay Buffer (if exists)")
    logger.info("=" * 80)

    replay_buffer_path = Path("./experiments/replay_buffer")

    if not replay_buffer_path.exists():
        logger.info(f"No actual replay buffer found at {replay_buffer_path}")
        logger.info("This is expected if no training has been completed yet")
        return True

    # Import the ExperienceReplayBuffer from training.py
    sys.path.insert(0, str(Path(__file__).parent))
    from training import ExperienceReplayBuffer

    try:
        buffer = ExperienceReplayBuffer.load(replay_buffer_path)

        if buffer is None:
            logger.warning(f"Could not load replay buffer from {replay_buffer_path}")
            return True

        stats = buffer.get_stats()
        logger.info(f"\nLoaded actual replay buffer:")
        logger.info(f"  Tasks: {stats['num_tasks']}")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Task names: {stats['task_names']}")

        # Check each task's samples
        required_fields = {"input_ids", "attention_mask", "answer"}

        for task_id, dataset in buffer.buffer.items():
            task_name = buffer.task_info[task_id]["task_name"]
            logger.info(f"\nTask {task_id} ({task_name}):")
            logger.info(f"  Samples: {len(dataset)}")
            logger.info(f"  Fields: {dataset.column_names}")

            fields = set(dataset.column_names)
            if required_fields.issubset(fields):
                logger.info(f"  ✓ All required fields present")

                # Sample check
                sample = dataset[0]
                logger.info(f"  Sample 0 answer: {sample['answer'][:50]}...")  # First 50 chars
            else:
                missing = required_fields - fields
                logger.error(f"  ✗ Missing fields: {missing}")
                return False

        logger.info("\n✓ Actual replay buffer verification passed")

    except Exception as e:
        logger.error(f"Error checking actual replay buffer: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = True

    # Test 1: Mock data test
    if not test_replay_buffer_fields():
        success = False

    # Test 2: Actual checkpoint test (if available)
    if not test_with_actual_checkpoint():
        success = False

    if success:
        logger.info("\n" + "🎉 ALL VERIFICATION PASSED 🎉")
        sys.exit(0)
    else:
        logger.error("\n" + "❌ SOME TESTS FAILED ❌")
        sys.exit(1)
