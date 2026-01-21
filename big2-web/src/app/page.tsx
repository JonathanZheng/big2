import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { createClient } from '@/lib/supabase/server';

export default async function Home() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-900 to-green-950 flex flex-col p-8">
      {/* Header */}
      <header className="flex justify-end mb-8">
        {user ? (
          <Link href="/profile">
            <Button variant="outline" className="border-green-400 text-green-100 hover:bg-green-800">
              Profile
            </Button>
          </Link>
        ) : (
          <div className="flex gap-2">
            <Link href="/auth/login">
              <Button variant="outline" className="border-green-400 text-green-100 hover:bg-green-800">
                Login
              </Button>
            </Link>
            <Link href="/auth/signup">
              <Button className="bg-green-600 hover:bg-green-500">
                Sign Up
              </Button>
            </Link>
          </div>
        )}
      </header>

      <div className="flex-1 flex flex-col items-center justify-center">
        <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold text-white mb-4">Big 2</h1>
          <p className="text-xl text-green-200">The classic Chinese card game</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Play vs Bots */}
          <Card className="bg-white/10 backdrop-blur border-green-400/30 hover:bg-white/20 transition-colors">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <span className="text-2xl">🤖</span>
                Play vs Bots
              </CardTitle>
              <CardDescription className="text-green-200">
                Practice against AI opponents
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-green-100 mb-4 text-sm">
                Play against 3 computer-controlled opponents. Perfect for learning or practicing strategies.
              </p>
              <Link href="/play/bot">
                <Button className="w-full bg-green-600 hover:bg-green-500">
                  Play Now
                </Button>
              </Link>
            </CardContent>
          </Card>

          {/* Online Multiplayer */}
          <Card className="bg-white/10 backdrop-blur border-green-400/30 hover:bg-white/20 transition-colors">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <span className="text-2xl">👥</span>
                Online Multiplayer
              </CardTitle>
              <CardDescription className="text-green-200">
                Play with friends or strangers
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-green-100 mb-4 text-sm">
                Create a room with a code for friends, or join the matchmaking queue to play with others.
              </p>
              <Link href="/play/online">
                <Button className="w-full bg-blue-600 hover:bg-blue-500">
                  Coming Soon
                </Button>
              </Link>
            </CardContent>
          </Card>

          {/* Test Mode */}
          <Card className="bg-white/10 backdrop-blur border-green-400/30 hover:bg-white/20 transition-colors">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <span className="text-2xl">🔬</span>
                Test Mode
              </CardTitle>
              <CardDescription className="text-green-200">
                Analyze real-life games
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-green-100 mb-4 text-sm">
                Enter your hand from a physical game and get AI suggestions for the best move.
              </p>
              <Link href="/play/test">
                <Button variant="outline" className="w-full border-green-400 text-green-100 hover:bg-green-800">
                  Coming Soon
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        <div className="mt-12 text-center text-green-300/60 text-sm">
          <p>Big 2 (锄大地) is a popular card game in East Asia</p>
        </div>
        </div>
      </div>
    </div>
  );
}
