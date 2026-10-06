# ABOUTME: A durable MemoryProvider built on Temporal itself: one long-running entity workflow per
# scope holds that scope's memories (see ``workflow.MemoryWorkflow``). Import the pieces from their
# own modules — this package __init__ stays empty because ``workflow`` is loaded inside the workflow
# sandbox and must not drag in ``service``/``provider`` (which hold a Temporal client).
#
#   worker:  service = MemoryService(client, task_queue=TASK_QUEUE)
#            configure_memory(TemporalMemoryProvider(service))
#            Worker(client, task_queue=TASK_QUEUE, workflows=[MemoryWorkflow, ...],
#                   activities=[embed_texts, ...])
