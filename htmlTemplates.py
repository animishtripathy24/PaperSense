css = '''
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    width: 100%;
}

.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    max-width: 90%;
}

.chat-message.user {
    background-color: #2b313e;
    align-self: flex-end;
}

.chat-message.bot {
    background-color: #475063;
    align-self: flex-start;
}

.chat-message .message {
    flex-grow: 1;
    color: #fff;
    padding: 0.5rem 1rem;
    font-size: 16px;
    word-wrap: break-word;
    max-width: 100%;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''