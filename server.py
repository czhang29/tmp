from flask import Flask, render_template, session  # Removed unused imports: request, time

app = Flask(__name__)

app.secret_key = 'replace-this-with-a-real-secret-key'

# Define lessons (Only include lessons 1 through 5)
lessons = {
    "1": {
        "title": "Introduction",
        "content": "This guide will help you achieve ideal meditation posture using real-time feedback.",
        "image": "images/intro_image.jpg"
    },
    "2": {
        "title": "Benefits & Disadvantages",
        "content": "Benefits of good posture and consequences of poor posture.",
        "image": "images/benefits_image.jpg"
    },
    "3": {
        "title": "Body Position Overview",
        "content": "Proper alignment creates a stable foundation for your practice.",
        # "image": "/static/images/body_map.png" # Silhouette is part of the template now
    },
    "4": {
        "title": "Head & Neck Position",
        "content": """
          <ul class="list-unstyled">
              <li><strong>Chin Tuck:</strong> Gently tuck your chin towards your throat, lengthening the back of your neck. Imagine a string pulling the crown of your head upwards.</li>
              <li><strong>Level Gaze:</strong> Keep your gaze soft and level, not looking up or down excessively.</li>
              <li><strong>Relaxed Jaw:</strong> Ensure your jaw is unclenched and your facial muscles are relaxed.</li>
          </ul>
        """,
        "image": "images/head_position_image.jpg"
    },
    "5": {
        "title": "Shoulders & Arms Position",
        "content": """
          <ul class="list-unstyled">
              <li><strong>Shoulders Down & Back:</strong> Gently roll your shoulders up, back, and then let them relax down, away from your ears. Avoid hunching.</li>
              <li><strong>Open Chest:</strong> Allow your chest to be open but relaxed, not puffed out rigidly.</li>
              <li><strong>Arm Placement:</strong> Rest your hands comfortably in your lap or on your knees, palms up or down, whichever feels more natural. Keep elbows slightly bent and relaxed.</li>
          </ul>
        """,
        "image": "images/shoulders_position_image.jpg"
    }
    # Lessons 6, 7, 8 are removed
}

# Define total steps for progress calculation
TOTAL_LEARNING_STEPS = 6 # 5 lessons + 1 practice page

@app.route('/')
def homepage():
    # FIX: Clear the learning flow flag when user returns to homepage
    session.pop('in_learning_flow', None) # Remove the key if it exists, do nothing otherwise
    progress_percent = 0
    return render_template('homepage.html', progress_percent=progress_percent)

@app.route('/learn/<lesson_number>')
def learn(lesson_number):
    lesson_data = lessons.get(lesson_number)
    if not lesson_data:
        # If lesson not found, clear flag and show 0 progress
        session.pop('in_learning_flow', None)
        return render_template('homepage.html', progress_percent=0), 404

    progress_percent = 0 # Default

    # Set flag at the start of learning
    if lesson_number == '1':
        session['in_learning_flow'] = True

    # Check if user is in the learning flow
    if session.get('in_learning_flow'):
        try:
            lesson_num_int = int(lesson_number)
            progress_percent = min(round((lesson_num_int / TOTAL_LEARNING_STEPS) * 100), 100)
        except ValueError:
            progress_percent = 0

    return render_template('learn.html',
                           lesson=lesson_data,
                           lesson_number=str(lesson_number),
                           lessons=lessons,
                           progress_percent=progress_percent)

@app.route('/practice')
def practice():
    progress_percent = 0 # Default

    # Show 100% progress ONLY if user was in the learning flow
    if session.get('in_learning_flow'):
        progress_percent = 100
    # NOTE: We don't clear the flag here, user might go back to lessons

    return render_template('practice.html', progress_percent=progress_percent)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)