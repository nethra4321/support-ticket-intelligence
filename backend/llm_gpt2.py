from transformers import pipeline

# Loads once at server start (slow first time)
_gen = pipeline("text-generation", model="gpt2")

def gpt2_summary_reply(ticket_text: str) -> tuple[str, str]:
    # GPT-2 isn't instruction tuned, so keep prompt simple
    prompt = (
        "Ticket:\n"
        f"{ticket_text}\n\n"
        "Summary:\n"
    )

    out = _gen(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=1,
    )[0]["generated_text"]

    # naive parsing: everything after "Summary:" is summary-ish
    summary = out.split("Summary:", 1)[-1].strip()
    # generate a separate reply prompt
    reply_prompt = (
        "Write a polite customer support reply to this ticket:\n"
        f"{ticket_text}\n\nReply:\n"
    )
    out2 = _gen(
        reply_prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=1,
    )[0]["generated_text"]
    reply = out2.split("Reply:", 1)[-1].strip()

    return summary[:400], reply[:600]
