# Replay Buffer Resume Verification

## Question
Does the replay buffer load correctly when resuming training?

## Resume Scenarios

### Scenario 1: Resume from Completed Task
Example: Tasks 0, 1 completed → Resume at task 2

**Code Path (line 866-876)**:
```python
# Resuming from a completed task checkpoint
start_task_id = last_task_id + 1  # 1 + 1 = 2
logger.info(f"Last completed task: {last_task_id} ({tasks[last_task_id]})")  # Task 1
logger.info(f"Resuming from task: {start_task_id}")  # Task 2
```

**Replay Buffer Loading (line 877-886)**:
```python
if self.use_experience_replay and self.replay_buffer is not None:
    loaded_buffer = ExperienceReplayBuffer.load(self.replay_buffer_dir)
    if loaded_buffer is not None:
        self.replay_buffer = loaded_buffer
        stats = self.replay_buffer.get_stats()
        logger.info(f"Replay buffer loaded: {stats['num_tasks']} tasks, "
                  f"{stats['total_samples']} samples")
```

**What's in the loaded buffer?**
- Task 0 samples (500 samples)
- Task 1 samples (500 samples)
- Total: 2 tasks, 1000 samples

**When training task 2**:
- `get_replay_dataset(current_task_id=2)` called (line 916)
- Returns samples from tasks 0-1 (range(2) = [0, 1])
- ✓ **CORRECT**: Task 2 trains with replay from tasks 0-1

---

### Scenario 2: Resume from Intermediate Checkpoint (NEW)
Example: Tasks 0, 1 completed → Task 2 partially trained → Crash → Resume

**Code Path (line 857-865)**:
```python
# Resuming from an intermediate checkpoint (partial task)
start_task_id = last_task_id  # Task 2
logger.info(f"Partial task: {last_task_id} ({tasks[last_task_id]})")  # Task 2
logger.info(f"Will continue training task: {start_task_id}")  # Task 2
```

**Replay Buffer Loading (line 877-886)**:
- Same loading code as Scenario 1
- Loads replay buffer from disk

**What's in the loaded buffer?**
- Task 0 samples (500 samples) ✓
- Task 1 samples (500 samples) ✓
- Task 2 samples? **NO** - Not added yet (task didn't complete)
- Total: 2 tasks, 1000 samples

**Critical Check**: Is task 2 in the buffer?
```python
# Replay buffer is saved AFTER checkpoint (line 816)
self.replay_buffer.save(self.replay_buffer_dir)

# But task 2 is added to buffer BEFORE checkpoint save (line 809-814)
self.replay_buffer.add_task_samples(
    task_id=task_id,
    dataset=tokenized_dataset["train"],
    task_name=task.dataset_name
)
```

**WAIT - Potential Issue Identified!**

Looking at the task loop (lines 906-945):
```
1. save_training_state(task_id, task_name)  # Line 908
2. Prepare dataset
3. Get replay dataset  # Line 916
4. Train model  # Line 919
5. Save checkpoint  # Line 927
6. Add to replay buffer  # Line 931
7. Save replay buffer  # Line 936
```

**Order of operations**:
- Train completes → Save checkpoint → **THEN** add to replay buffer

**If crash happens AFTER save_checkpoint but BEFORE replay buffer save**:
- Task checkpoint saved to `continual/`
- Replay buffer NOT updated yet
- On resume: Will see completed task, but replay buffer missing that task!

**Let me trace through this scenario**:

1. Task 1 completes training
2. Checkpoint saved to `continual/task_1_FOMC`
3. **CRASH before `add_task_samples`**
4. Resume:
   - Finds `task_1_FOMC` checkpoint
   - Loads replay buffer (only has task 0)
   - Starts task 2
   - Gets replay from task 0 only (missing task 1!)

**THIS IS A BUG!** 🐛

---

## Issue Summary

**Problem**: Replay buffer update happens AFTER checkpoint save, creating a window where:
- Task appears completed (checkpoint exists)
- But replay buffer doesn't have that task's samples

**Impact**:
- If crash occurs between checkpoint save and replay buffer save
- Next tasks won't get replay samples from the incomplete task
- Increases catastrophic forgetting

**Current Code (lines 927-936)**:
```python
# Save checkpoint
self.save_checkpoint(task_id, task.dataset_name)

# Add samples from current task to replay buffer for future tasks
if self.use_experience_replay and self.replay_buffer is not None:
    self.replay_buffer.add_task_samples(
        task_id=task_id,
        dataset=tokenized_dataset["train"],
        task_name=task.dataset_name
    )
    # Save replay buffer after each task
    self.replay_buffer.save(self.replay_buffer_dir)
```

---

## Proposed Fix

**Option 1: Add to replay buffer BEFORE checkpoint save**
```python
# Add samples to replay buffer FIRST
if self.use_experience_replay and self.replay_buffer is not None:
    self.replay_buffer.add_task_samples(
        task_id=task_id,
        dataset=tokenized_dataset["train"],
        task_name=task.dataset_name
    )
    self.replay_buffer.save(self.replay_buffer_dir)

# THEN save checkpoint (marks task as complete)
self.save_checkpoint(task_id, task.dataset_name)
```

**Pros**:
- ✓ Replay buffer always has samples from "completed" tasks
- ✓ Safer ordering

**Cons**:
- If crash after replay save but before checkpoint save:
  - Replay buffer has task N samples
  - But task N checkpoint doesn't exist
  - On resume: start_task_id = N, get_replay_dataset(N) returns tasks 0 to N-1
  - Task N samples in buffer are unused until task N completes
  - Minor inefficiency but not incorrect

**Option 2: Atomic save with validation on load**
- Save both together
- On load, validate replay buffer matches checkpoints
- Rebuild missing entries

---

## Recommendation

**Use Option 1**: Add to replay buffer BEFORE checkpoint save

**Rationale**:
1. Safer failure mode (extra samples in buffer is benign)
2. Simpler implementation
3. No data loss
4. Replay buffer always consistent with completed checkpoints

**Implementation**: Swap lines 927-936 order
