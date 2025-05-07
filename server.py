from flask import Flask, render_template, session, redirect, url_for, request, jsonify

app = Flask(__name__)

app.secret_key = 'replace-this-with-a-real-secret-key'

# Define lessons (Only include lessons 1 through 5, removed spine lesson)
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
}

# Define total steps for progress calculation (reduced by 1 since we removed spine)
TOTAL_LEARNING_STEPS = 6 # 5 lessons + 1 practice page

@app.route('/')
def homepage():
    # Clear the learning flow flag when user returns to homepage
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
    # Redirect to timer selection page instead of going directly to practice
    return redirect(url_for('timer_selection', practice_focus='all'))

@app.route('/practice/head_neck')
def practice_head_neck():
    progress_percent = 0
    if session.get('in_learning_flow'):
        progress_percent = min(round((4 / TOTAL_LEARNING_STEPS) * 100), 100)
    
    session['practice_focus'] = 'head_neck'
    
    # Redirect to timer selection page
    return redirect(url_for('timer_selection', practice_focus='head_neck'))

@app.route('/practice/shoulders')
def practice_shoulders():
    progress_percent = 0
    if session.get('in_learning_flow'):
        progress_percent = min(round((5 / TOTAL_LEARNING_STEPS) * 100), 100)
    
    session['practice_focus'] = 'shoulders'
    
    # Redirect to timer selection page
    return redirect(url_for('timer_selection', practice_focus='shoulders'))

@app.route('/timer_selection')
def timer_selection():
    """Show timer selection page"""
    practice_focus = request.args.get('practice_focus', 'all')
    session['practice_focus'] = practice_focus
    
    progress_percent = 0
    if session.get('in_learning_flow'):
        if practice_focus == 'head_neck':
            progress_percent = min(round((4 / TOTAL_LEARNING_STEPS) * 100), 100)
        elif practice_focus == 'shoulders':
            progress_percent = min(round((5 / TOTAL_LEARNING_STEPS) * 100), 100)
        else:
            progress_percent = 100
    
    return render_template('timer_selection.html', progress_percent=progress_percent)

@app.route('/start_countdown', methods=['POST'])
def start_countdown():
    """Process timer selection and show countdown page"""
    if request.method == 'POST':
        # Get practice duration from form
        practice_duration_seconds = request.form.get('practice_duration_seconds', '120')
        practice_focus = request.form.get('practice_focus', 'all')
        
        # Store in session
        session['practice_duration_seconds'] = int(practice_duration_seconds)
        session['practice_focus'] = practice_focus
        
        # Determine redirect URL based on practice focus
        if practice_focus == 'head_neck':
            redirect_url = url_for('practice_head_neck_session')
        elif practice_focus == 'shoulders':
            redirect_url = url_for('practice_shoulders_session')
        else:
            redirect_url = url_for('practice_session')
        
        return render_template('countdown.html', redirect_url=redirect_url)
    
    # If not POST, redirect to selection page
    return redirect(url_for('timer_selection'))

@app.route('/practice_session')
def practice_session():
    """Show the actual practice page after countdown"""
    progress_percent = 0
    
    # Show 100% progress ONLY if user was in the learning flow
    if session.get('in_learning_flow'):
        progress_percent = 100
    
    # Get practice duration from session
    practice_duration_seconds = session.get('practice_duration_seconds', 120)
    
    return render_template('practice.html', 
                           progress_percent=progress_percent, 
                           practice_duration_seconds=practice_duration_seconds)

@app.route('/practice_head_neck_session')
def practice_head_neck_session():
    """Show focused practice page for head & neck after countdown"""
    progress_percent = 0
    if session.get('in_learning_flow'):
        progress_percent = min(round((4 / TOTAL_LEARNING_STEPS) * 100), 100)
    
    # Get practice duration from session
    practice_duration_seconds = session.get('practice_duration_seconds', 120)
    
    return render_template('practice_focused.html', 
                           focus="Head & Neck", 
                           title="Head & Neck Practice", 
                           progress_percent=progress_percent,
                           practice_duration_seconds=practice_duration_seconds)

@app.route('/practice_shoulders_session')
def practice_shoulders_session():
    """Show focused practice page for shoulders after countdown"""
    progress_percent = 0
    if session.get('in_learning_flow'):
        progress_percent = min(round((5 / TOTAL_LEARNING_STEPS) * 100), 100)
    
    # Get practice duration from session
    practice_duration_seconds = session.get('practice_duration_seconds', 120)
    
    return render_template('practice_focused.html', 
                           focus="Shoulders", 
                           title="Shoulders Practice", 
                           progress_percent=progress_percent,
                           practice_duration_seconds=practice_duration_seconds)

@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    """API endpoint to save practice feedback data"""
    if request.method == 'POST':
        try:
            # Get feedback data from request
            feedback_data = request.get_json()
            
            # Store in session
            for key, value in feedback_data.items():
                session[key] = value
                
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@app.route('/feedback_summary')
def feedback_summary():
    """Show practice feedback summary"""
    # Default values if session doesn't have data
    if 'head_neck_score' not in session:
        # Get practice focus (default to 'all' if not set)
        focus = session.get('practice_focus', 'all')
        
        # Set default scores based on focus
        if focus == 'head_neck':
            session['head_neck_score'] = 75
            session['overall_score'] = 75
            session['head_neck_feedback'] = "Your head and neck position is good, but could use some improvement. Try to keep your chin slightly tucked and ears level."
        elif focus == 'shoulders':
            session['shoulders_score'] = 60
            session['overall_score'] = 60
            session['shoulders_feedback'] = "Your shoulders show some tension. Practice relaxing them down and back, keeping them level with each other."
        else:
            # Full practice
            session['head_neck_score'] = 75
            session['shoulders_score'] = 60
            session['overall_score'] = 68  # Average of head/neck and shoulders
            session['head_neck_feedback'] = "Your head and neck position is good, but could use some improvement. Try to keep your chin slightly tucked and ears level."
            session['shoulders_feedback'] = "Your shoulders show some tension. Practice relaxing them down and back, keeping them level with each other."
        
        # Get practice duration from session (default to 2 minutes)
        practice_duration_seconds = session.get('practice_duration_seconds', 120)
        minutes = practice_duration_seconds // 60
        seconds = practice_duration_seconds % 60
        session['practice_duration'] = f"{minutes}:{seconds:02d}"
        
        session['overall_feedback'] = "Your overall posture is good with some areas that could use improvement. With regular practice, you'll develop better postural habits."
    
    return render_template('feedback_summary.html')

@app.route('/results')
def results():
    """Show detailed results page"""
    # Ensure we have basic data for results
    if 'head_neck_score' not in session:
        # If no scores are set, redirect to summary to initialize defaults
        return redirect(url_for('feedback_summary'))
    
    # Get practice focus
    focus = session.get('practice_focus', 'all')
    
    # For demo, add more detailed metrics if not present
    if 'head_alignment' not in session:
        # Set default metrics based on focus
        if focus == 'head_neck' or focus == 'all':
            # Head & Neck details
            session['head_alignment'] = 'Good'
            session['head_alignment_score'] = 70
            session['neck_tension'] = 'Slightly Tense'
            session['neck_tension_score'] = 65
            session['gaze_direction'] = 'Level'
            session['gaze_direction_score'] = 80
            session['head_neck_recommendation'] = 'Practice daily chin tucks to strengthen your neck and improve alignment.'
        
        if focus == 'shoulders' or focus == 'all':
            # Shoulders details
            session['shoulder_level'] = 'Slightly Uneven'
            session['shoulder_level_score'] = 60
            session['shoulder_tension'] = 'Tense'
            session['shoulder_tension_score'] = 50
            session['chest_openness'] = 'Partially Open'
            session['chest_openness_score'] = 65
            session['shoulders_recommendation'] = 'Perform shoulder rolls and gentle stretches throughout the day to release tension.'
        
        # Overall plan recommendations
        session['recommendation_1'] = 'Practice 5 minutes of posture awareness at your desk'
        session['recommendation_2'] = 'Do 2-3 gentle neck stretches every hour'
        session['recommendation_3'] = 'Strengthen your core with 10 minutes of exercises daily'
        session['overall_recommendation'] = 'Based on your assessment, we recommend the following daily practice:'
        session['recommendation_footer'] = 'Implement these practices consistently for 2 weeks, then return for another assessment to track your progress.'
    
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)